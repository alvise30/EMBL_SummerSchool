import argparse
import concurrent
import GPUtil
import logging.config
import os
import pbs3
import re
import signal
import subprocess
import sys
import time

from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED, Future
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Set, List, Deque, Tuple, Pattern

CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s.%(msecs)03d [%(processName)s/%(threadName)s] %(levelname)s %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "default",
        }
    },
    "loggers": {
        "": {"handlers": ["default"], "level": "WARNING", "propagate": True},
        "__main__": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        "summer": {"handlers": ["default"], "level": "WARNING", "propagate": False},
    },
}
logging.config.dictConfig(CONFIG)
logger = logging.getLogger(__name__)


CUDA_VISIBLE_DEVICES_NAME = "CUDA_VISIBLE_DEVICES"
PYTHON_PATH_NAME = "PYTHONPATH"
SUCCEEDED_ON = "succeeded_on_"
FAILED_ON = "failed_on_"


class FreeGPUTracker:
    previous = []

    def __call__(self) -> List[int]:
        """
        :return: available gpu ids
        """
        now = GPUtil.getAvailable(
            order="last", limit=100, maxLoad=0.01, maxMemory=0.01, includeNan=False, excludeID=[], excludeUUID=[]
        )
        both = [n for n in now if n in self.previous]
        self.previous = list(now)
        return both or now


def _run_in_clean_lab(remote_url: str, add_in_name: str, full_hash: str, cuda_id: int) -> None:
    """
    note: using `git --git-dir tmp_repo_dir` in order not to change global working directory. (Alternatively subprocess
          might be used, but concurrent.futures.ProcessPoolExecutor apparently shares the working directory across
          processes)

    :param remote_url: of git repo
    :param full_hash: 'git commit hash to be run from'
    """
    with TemporaryDirectory() as lab:
        os.chdir(lab)
        logger.debug("running in clean lab: %s", lab)
        try:
            repo_name = f"summer_{full_hash}"
            pbs3.git.clone("--recurse-submodules", "-j8", remote_url, repo_name)
            os.chdir(repo_name)
            pbs3.git("fetch", "origin", full_hash)
            pbs3.git("checkout", "--force", full_hash)
            pbs3.git("submodule", "update", "--recursive")

            test_env = os.environ.copy()
            test_env[PYTHON_PATH_NAME] = os.path.join(lab, repo_name) + os.pathsep + test_env.get(PYTHON_PATH_NAME, "")
            test_env[CUDA_VISIBLE_DEVICES_NAME] = str(cuda_id)

            cmd = ["python", "-c", f'import summer;summer.run("{add_in_name}")']
            out = subprocess.run(
                cmd, env=test_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
            )
            if out.returncode:
                tag = f"{FAILED_ON}{datetime.now().strftime('%y-%m-%d_%H-%M')}"
            else:
                tag = f"{SUCCEEDED_ON}{datetime.now().strftime('%y-%m-%d_%H-%M')}"

            msg = out.stdout.decode("utf-8").replace("'", '"')
            pbs3.git("tag", "-a", tag, "-m", f"'{msg}'", full_hash)
            pbs3.git("push", "origin", tag)
            logger.debug("pushed tag '%s' to '%s' at %s", tag, "origin", full_hash)
        except Exception as e:
            logger.exception(e)
            raise e


def experimenter(
    start_commit: str,
    remote_branches: List[Pattern[str]],
    experiment_identifier: Pattern[str],
    rerun_succeeded: bool,
    rerun_failed: bool,
):
    """
    :param start_commit: exclusive start commit-ish identifier (commit hash, tag, or branch)
    :param remote_branches: specifies which remote branches or commit-ish objects to track.
                            Glob like git patterns are valid specifiers as well
    :param experiment_identifier: a commit is considered an experiment, if the commit subject matches 'experiment identifier'
    :param rerun_succeeded: whether or not to rerun previously succeeded experiments
    :param rerun_failed: whether or not to rerun previously failed experiments
    :return:
    """
    remote_url = pbs3.git.remote("get-url", remote_branches[0].split("/")[0]).strip()
    assert not remote_url.startswith("http"), "use ssh address 'git@...'"

    cuda_devices = os.environ.get(CUDA_VISIBLE_DEVICES_NAME, None)
    if cuda_devices is None:
        raise Exception(f"{CUDA_VISIBLE_DEVICES_NAME} not set")

    cuda_ids = set(map(int, cuda_devices.split(",")))
    n_gpus = len(cuda_ids)
    cuda_ids_externally_busy: Set[int] = set()
    commits_to_run: Deque[Tuple[str, str]] = deque()

    def ignore_sigint():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    free_gpu_tracker = FreeGPUTracker()

    with TemporaryDirectory() as repo_dir, ProcessPoolExecutor(
        max_workers=n_gpus, initializer=ignore_sigint
    ) as executor:
        logger.debug("working in %s", repo_dir)
        os.chdir(repo_dir)
        pbs3.git.clone("--recurse-submodules", "-j8", remote_url, "summer_experimenter")
        os.chdir("summer_experimenter")

        futs: Set[Future] = set()
        seen_exp_commits: Set[str] = set()
        split_at = "git-full-hash-left-and-git-subject-right"

        active = datetime.now()
        seen_remote_branches: List[str] = []
        matching_remote_branches: List[str] = []
        non_matching_remote_branches: List[str] = []
        try:
            while active > datetime.now() - timedelta(days=1):  # limit waiting time to one day
                # check for new remote branches
                for remote in set(rbn.split("/")[0] for rbn in remote_branches):
                    logger.debug("git fetch %s", remote)
                    pbs3.git.fetch(remote)

                new_remote_branches = filter(
                    lambda cr: cr not in seen_remote_branches, pbs3.git.branch("-r").split("\n")
                )
                for rbn in new_remote_branches:
                    seen_remote_branches.append(rbn)
                    if "->" in rbn:
                        rbn = rbn.split("->")[1]

                    rbn = rbn.strip()

                    for bn_regex in remote_branches:
                        if re.fullmatch(bn_regex, rbn):
                            logger.debug("found matching remote branch: %s", rbn)
                            matching_remote_branches.append(rbn)
                        else:
                            logger.debug("found non-matching remote branch: %s", rbn)
                            non_matching_remote_branches.append(rbn)

                # check for new commits in matching remote branches
                for bn in matching_remote_branches:
                    # git log --ancestry-path HEAD^..bn --pretty=tformat:%H-split_at-%s --committer=fynnbe@gmail.com -n 100
                    next_commits_cmd = pbs3.git.log(
                        "--ancestry-path",
                        f"{start_commit}..{bn}",
                        f"--pretty=tformat:%H{split_at}%s",
                        "--committer=fynnbe@gmail.com",
                        "-n",
                        100,
                        # _tty_out=False,  # suppresses escape sequence (needed when using sh)
                    )
                    next_commits: List[str] = next_commits_cmd.stdout.strip().split("\n")
                    next_commits = list(filter(lambda nc: nc, next_commits))
                    for nc in next_commits:
                        logger.debug("nc: %s", nc)
                        full_hash, subject = nc.split(split_at)
                        if not re.match(experiment_identifier, subject):
                            logger.debug("skip non-experiment commit: %s %s", full_hash[:7], subject)
                        elif full_hash in seen_exp_commits:
                            logger.debug("skip previously encountered experiment commit: %s %s", full_hash[:7], subject)
                        else:
                            # check if commit has been run previously
                            try:
                                tags = pbs3.git.describe("--tags", "--exact-match", full_hash)
                            except pbs3.ErrorReturnCode_128:
                                tags = ""
                                logger.debug("no tags found at %s", full_hash)

                            if not rerun_succeeded and SUCCEEDED_ON in tags or not rerun_failed and FAILED_ON in tags:
                                logger.debug("Not rerunning '%s' with tags: %s", full_hash, tags)
                                continue

                            seen_exp_commits.add(full_hash)
                            commits_to_run.append((bn, full_hash))

                if not commits_to_run:
                    # take a break
                    logger.debug("sleep")
                    time.sleep(60)
                else:
                    logger.debug("commits waiting to run:\n%s", "".join(["\t" + str(c) for c in commits_to_run]))

                # submit to all available gpus
                while commits_to_run and cuda_ids:
                    cuda_id = cuda_ids.pop()
                    # check if popped cuda id is used elsewhere
                    if cuda_id not in free_gpu_tracker():
                        cuda_ids_externally_busy.add(cuda_id)
                        continue

                    bn, full_hash = commits_to_run.popleft()
                    remote_url = pbs3.git.remote("get-url", bn.split("/")[0]).strip()
                    assert not remote_url.startswith("http"), "use ssh address 'git@...'"
                    print(f"submit for gpu {cuda_id}")
                    a_fut = executor.submit(
                        _run_in_clean_lab,
                        remote_url=remote_url,
                        add_in_name=bn.split("/")[1],
                        full_hash=full_hash,
                        cuda_id=cuda_id,
                    )
                    a_fut.cuda_id = cuda_id
                    futs.add(a_fut)
                    active = datetime.now()

                # wait for at least one available gpu
                if not cuda_ids:
                    # check if externally used gpus are still busy
                    if cuda_ids_externally_busy:
                        available = free_gpu_tracker()
                        for cuda_id in set(cuda_ids_externally_busy):
                            if cuda_id in available:
                                cuda_ids_externally_busy.remove(cuda_id)
                                cuda_ids.add(cuda_id)

                    try:
                        # wait at most half an hour (to check again on externally busy gpus)
                        done_futs, futs = wait(futs, return_when=FIRST_COMPLETED, timeout=30 * 60)
                        cuda_ids |= set(map(lambda df: df.cuda_id, done_futs))
                        active = datetime.now()
                    except concurrent.futures.TimeoutError:
                        pass

        except (Exception, KeyboardInterrupt, SystemExit) as e:
            logger.exception(e)
            print("Waiting for running experiments to finish")
            concurrent.futures.wait(futs)
            executor.shutdown(wait=True)  # this alone is somehow not sufficient, added wait(futs) above...
            for fut in futs:
                fute = fut.exception()
                if fute is not None:
                    logger.exception(fute)

            print("All experiments done or not started")


if __name__ == "__main__":
    assert "summer" not in sys.modules, "summer must not be imported previously!"
    assert "torch" not in sys.modules, "pytorch must not be imported previously!"

    DATA_FOLDER = "DATA_FOLDER"
    data_folder = os.environ.get(DATA_FOLDER, None)
    if data_folder is None:
        data_folder = Path(__file__).parent.parent / "data"
        data_folder.mkdir(exist_ok=True)
        os.environ[DATA_FOLDER] = data_folder.resolve().as_posix()

    LOG_DIR = "LOG_DIR"
    log_dir = os.environ.get(LOG_DIR, None)
    if log_dir is None:
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        os.environ[LOG_DIR] = log_dir.resolve().as_posix()

    parser = argparse.ArgumentParser(description="summer experimenter")
    parser.add_argument(
        "remote_branch",
        metavar="remote_branch",
        type=str,
        nargs="+",
        help="specifies which remote branches or commit-ish objects to track. "
        "Regex patterns are evaluated for a fullmatch.",
    )
    parser.add_argument(
        "--exp",
        metavar="experiment identifier",
        type=str,
        default="exp:",
        help="a commit is considered an experiment, if the commit subject matches 'experiment identifier'",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="HEAD^",
        help="exlusive start commit-ish identifier (commit hash, tag, or branch).\n"
        "Default: running experiments from current HEAD inclusively (as HEAD^ is the parent of HEAD )",
    )
    parser.add_argument(
        "--cuda", type=str, default=None, help="comma separated cuda ids to run on (CUDA_VISIBLE_DEVICES)"
    )
    parser.add_argument(
        "--rerun-failed", action="store_true", help="wether or not to rerun previously failed experiments"
    )
    parser.add_argument(
        "--rerun-succeeded", action="store_true", help="whether or not to rerun previously succeeded experiments"
    )
    args = parser.parse_args()

    cuda_devices = args.cuda
    if cuda_devices is None:
        cuda_devices = os.environ.get(CUDA_VISIBLE_DEVICES_NAME, None)

    if cuda_devices is None:
        raise Exception(f"{CUDA_VISIBLE_DEVICES_NAME} not set")

    commit_hash_cmd = pbs3.git("rev-parse", "--verify", args.start)
    current_commit_hash = commit_hash_cmd.stdout

    remote_branches = [bn if "/" in bn else "origin/" + bn for bn in args.remote_branch]
    experimenter(
        start_commit=current_commit_hash.strip(),
        experiment_identifier=args.exp,
        remote_branches=remote_branches,
        rerun_failed=args.rerun_failed,
        rerun_succeeded=args.rerun_succeeded,
    )
