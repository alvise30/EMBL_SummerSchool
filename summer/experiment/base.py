import logging
from enum import Enum

import matplotlib.pyplot as plt
import numpy
import os
import pbs3
import torch.nn
import torchvision.transforms.functional as TF

from dataclasses import dataclass
from datetime import datetime
from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, EarlyStopping, TerminateOnNan
from ignite.metrics import Loss, Accuracy
from ignite.utils import convert_tensor
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from PIL import Image
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable, Type, Any, Tuple, Sequence

from summer.datasets import CellTrackingChallengeDataset
from summer.utils.stat import DatasetStat

eps_for_precision = {torch.half: 1e-4, torch.float: 1e-8}

IN_TRAINING = "in_training"
TRAINING = "training"
VALIDATION = "validation"
TEST = "test"

X_NAME = "x"
Y_NAME = "y"
Y_PRED_NAME = "y_pred"
LOSS_NAME = "Loss"
ACCURACY_NAME = "Accuracy"


@dataclass
class LogConfig:
    log_dir: Path = Path(os.environ.get("LOG_DIR", Path(__name__).parent.parent.parent / "logs"))

    validate_every_nth_epoch: int = 1
    log_scalars_every: Tuple[int, str] = (1, "iterations")
    log_images_every: Tuple[int, str] = (1, "epochs")


class ExperimentBase:
    model: torch.nn.Module
    depth: int

    train_dataset: CellTrackingChallengeDataset
    valid_dataset: CellTrackingChallengeDataset
    test_dataset: Optional[CellTrackingChallengeDataset]
    max_validation_samples: int
    only_eval_where_true: bool

    batch_size: int
    eval_batch_size: int
    precision: torch.dtype
    loss_fn: torch.nn.Module
    optimizer_cls: Type[torch.optim.Optimizer]
    optimizer_kwargs: Dict[str, Any]
    max_num_epochs: int

    model_checkpoint: Optional[Path]

    score_function: Callable[[Engine], float]

    add_in_name: Optional[str] = None

    def __init__(self):
        self.log_config = LogConfig()
        assert self.log_config.log_scalars_every[1] in ("iterations", "epochs"), self.log_config.log_scalars_every[1]
        assert self.log_config.log_images_every[1] in ("iterations", "epochs"), self.log_config.log_images_every[1]

        self.commit_hash = pbs3.git("rev-parse", "--verify", "HEAD").stdout
        self.commit_subject = pbs3.git.log("-1", "--pretty=%B").stdout.split("\n")[0]
        if self.add_in_name is None:
            self.add_in_name = (
                pbs3.git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip().replace("'", "").replace('"', "")
            )
        if self.valid_dataset == self.test_dataset and self.max_validation_samples >= len(self.test_dataset):
            raise ValueError("no samples for testing left")

    def to_tensor(self, img: Image.Image, seg: Image.Image, stat: DatasetStat) -> Tuple[torch.Tensor, torch.Tensor]:
        img: torch.Tensor = TF.to_tensor(img)
        seg: torch.Tensor = TF.to_tensor(seg)
        assert img.shape == seg.shape, (img.shape, seg.shape)
        assert seg.shape[0] == 1, seg.shape  # assuming singleton channel axis
        cut1 = img.shape[1] % 2 ** self.depth
        if cut1:
            img = img[:, cut1 // 2 : -((cut1 + 1) // 2)]
            seg = seg[:, cut1 // 2 : -((cut1 + 1) // 2)]

        cut2 = img.shape[2] % 2 ** self.depth
        if cut2:
            img = img[:, :, cut2 // 2 : -((cut2 + 1) // 2)]
            seg = seg[:, :, cut2 // 2 : -((cut2 + 1) // 2)]

        img = img.clamp(stat.x_min, stat.x_max)
        img = TF.normalize(img, mean=[stat.x_mean], std=[stat.x_std])

        return img.to(dtype=self.precision), (seg[0] != 0).to(dtype=self.precision)

    def test(self):
        self.max_num_epochs = 0
        self.run()

    def run(self):
        short_commit_subject = (
            self.commit_subject[5:15].replace(":", "").replace("'", "").replace('"', "").replace(" ", "_")
        )
        self.name = f"{datetime.now().strftime('%y-%m-%d_%H-%M')}_{self.commit_hash[:7]}_{self.add_in_name}_{short_commit_subject}"
        self.logger = logging.getLogger(self.name)
        self.log_config.log_dir /= self.name
        self.log_config.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_config.log_dir / "commit_hash").open("w") as f:
            f.write(self.commit_hash)

        devices = list(range(torch.cuda.device_count()))
        if devices:
            device = torch.device("cuda", devices[0])
        else:
            device = torch.device("cpu")

        self.model = self.model.to(device=device, dtype=self.precision)
        if self.model_checkpoint is not None:
            self.model.load_state_dict(torch.load(self.model_checkpoint, map_location=device))

        optimizer = self.optimizer_cls(self.model.parameters(), **self.optimizer_kwargs)
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, pin_memory=True, num_workers=16, shuffle=True
        )
        train_loader_eval = DataLoader(
            self.train_dataset,
            batch_size=self.eval_batch_size,
            pin_memory=True,
            num_workers=16,
            sampler=SubsetSequentialSampler(range(min(200, len(self.train_dataset)))),
        )
        valid_loader = (
            None
            if self.valid_dataset is None
            else DataLoader(
                self.valid_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=SubsetSequentialSampler(range(min(self.max_validation_samples, len(self.valid_dataset)))),
            )
        )
        if self.valid_dataset == self.test_dataset:
            test_sampler = SubsetSequentialSampler(range(self.max_validation_samples, len(self.test_dataset)))
        else:
            test_sampler = SubsetSequentialSampler(range(len(self.test_dataset)))

        test_loader = (
            None
            if self.test_dataset is None
            else DataLoader(
                self.test_dataset,
                batch_size=self.eval_batch_size,
                pin_memory=True,
                num_workers=16,
                sampler=test_sampler,
            )
        )

        # tensorboardX
        writer = SummaryWriter(log_dir=self.log_config.log_dir.as_posix())
        # x, y = self.train_dataset[0]
        # try:
        #     model_device = next(self.model.parameters(True)).get_device()
        #     if model_device >= 0:
        #         x = x.to(device=model_device)
        #     writer.add_graph(self.model, x)
        # except Exception as e:
        #     self.logger.warning("Failed to save model graph...")
        #     # self.logger.exception(e)

        # ignite
        class CustomEvents(Enum):
            VALIDATION_DONE = "validation_done_event"

        def training_step(engine, batch):
            self.model.train()
            optimizer.zero_grad()
            x, y = batch
            x = convert_tensor(x, device=device, non_blocking=False)
            y = convert_tensor(y, device=device, non_blocking=False)
            y_pred = self.model(x)
            assert len(y_pred.shape) == 4, "assuming 4dim model output: NCHW"
            assert y_pred.shape[1] == 1, "assuming singleton channel axis in model output"
            y_pred = y_pred.squeeze(dim=1)
            loss = self.loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            return {X_NAME: x, Y_NAME: y, Y_PRED_NAME: y_pred, LOSS_NAME: loss}

        trainer = Engine(training_step)
        trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

        def inference_step(engine, batch):
            self.model.eval()
            with torch.no_grad():
                x, y = batch
                x: torch.Tensor = convert_tensor(x, device=device, non_blocking=False)
                y: torch.Tensor = convert_tensor(y, device=device, non_blocking=False)
                y_pred = self.model(x)
                assert len(y_pred.shape) == 4, "assuming 4dim model output: NCHW"
                assert y_pred.shape[1] == 1, "assuming singleton channel axis in model output"
                y_pred = y_pred.squeeze(dim=1)
                if self.only_eval_where_true:
                    mask = y.eq(1)
                else:
                    mask = ...

                loss = self.loss_fn(y_pred[mask], y[mask])
                return {X_NAME: x, Y_NAME: y, Y_PRED_NAME: y_pred, LOSS_NAME: loss}

        class EngineWithMode(Engine):
            def __init__(self, process_function, modes: Sequence[str]):
                super().__init__(process_function=process_function)
                self.modes = modes
                self._mode = None

            @property
            def mode(self) -> str:
                if self._mode is None:
                    raise RuntimeError("mode not set")

                return self._mode

            @mode.setter
            def mode(self, new_mode: str):
                if new_mode not in self.modes:
                    raise ValueError(new_mode)
                else:
                    self._mode = new_mode

        evaluator = EngineWithMode(inference_step, modes=[TRAINING, VALIDATION, TEST])
        evaluator.register_events(*CustomEvents)
        evaluator.mode = TRAINING
        saver = ModelCheckpoint(
            (self.log_config.log_dir / "models").as_posix(),
            "v0",
            score_function=self.score_function,
            n_saved=1,
            create_dir=True,
            save_as_state_dict=True,
        )
        evaluator.add_event_handler(CustomEvents.VALIDATION_DONE, saver, {"models": self.model})
        stopper = EarlyStopping(patience=10, score_function=self.score_function, trainer=trainer)
        evaluator.add_event_handler(CustomEvents.VALIDATION_DONE, stopper)

        Loss(loss_fn=lambda loss, _: loss, output_transform=lambda out: (out[LOSS_NAME], out[X_NAME])).attach(
            evaluator, LOSS_NAME
        )
        Accuracy(output_transform=lambda out: (out[Y_PRED_NAME] > 0, out[Y_NAME]), is_multilabel=False).attach(
            evaluator, ACCURACY_NAME
        )

        result_saver = ResultSaver(TEST, file_path=self.log_config.log_dir / "results")

        @evaluator.on(Events.ITERATION_COMPLETED)
        def export_result(engine: EngineWithMode):
            result_saver.save(engine.mode, batch=engine.state.output[Y_PRED_NAME], at=engine.state.iteration - 1)

        def log_images(engine: Engine, name: str, step: int):
            x_batch = engine.state.output[X_NAME].cpu().numpy()
            y_batch = engine.state.output[Y_NAME].cpu().numpy()
            y_pred_batch = engine.state.output[Y_PRED_NAME].detach().cpu().numpy()
            assert x_batch.shape[0] == y_batch.shape[0], (x_batch.shape, y_batch.shape)
            assert len(y_batch.shape) == 3, y_batch.shape

            fig, ax = plt.subplots(
                nrows=x_batch.shape[0], ncols=4, squeeze=False, figsize=(4 * 3, x_batch.shape[0] * 3)
            )
            fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)

            def make_subplot(ax, img, cb=True):
                im = ax.imshow(img.astype(numpy.float))
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.axis("off")
                if cb:
                    # from https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
                    # create an axes on the right side of ax. The width of cax will be 5%
                    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="3%", pad=0.03)
                    fig.colorbar(im, cax=cax)

            for i, (xx, yy, pp) in enumerate(zip(x_batch, y_batch, y_pred_batch)):
                if i == 0:
                    ax[0, 0].set_title("input")
                    ax[0, 1].set_title("target")
                    ax[0, 2].set_title("output")
                    ax[0, 3].set_title("correct")

                make_subplot(ax[i, 0], xx[0])
                make_subplot(ax[i, 1], yy)
                pp_prob = 1 / (1 + numpy.exp(-pp))
                make_subplot(ax[i, 2], pp_prob)
                pp = pp > 0
                correct = numpy.equal(pp, yy)
                if self.only_eval_where_true:
                    mask = numpy.logical_not(yy)
                else:
                    mask = numpy.zeros_like(pp)

                wrong = numpy.logical_not(numpy.logical_or(correct, mask))
                make_subplot(ax[i, 3], numpy.stack([wrong, correct, mask], axis=-1), cb=False)

            plt.tight_layout()

            writer.add_figure(f"{name}/in_out", fig, step)

        def log_eval(engine: Engine, name: str, step: int):
            met = engine.state.metrics
            writer.add_scalar(f"{name}/{LOSS_NAME}", met[LOSS_NAME], step)
            writer.add_scalar(f"{name}/{ACCURACY_NAME}", met[ACCURACY_NAME], step)
            log_images(engine=engine, name=name, step=step)

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_training_iteration(engine: Engine):
            i = (engine.state.iteration - 1) % len(train_loader)
            if self.log_config.log_scalars_every[1] == "iterations" and i % self.log_config.log_scalars_every[0] == 0:
                writer.add_scalar(
                    f"{IN_TRAINING}/{LOSS_NAME}", engine.state.output[LOSS_NAME].item(), engine.state.iteration
                )

            if self.log_config.log_images_every[1] == "iterations" and i % self.log_config.log_images_every[0] == 0:
                log_images(engine, IN_TRAINING, engine.state.iteration)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_epoch(engine: Engine):
            epoch = engine.state.epoch
            if self.log_config.log_scalars_every[1] == "epochs" and epoch % self.log_config.log_scalars_every[0] == 0:
                writer.add_scalar(f"{IN_TRAINING}/{LOSS_NAME}", engine.state.output[LOSS_NAME].item(), epoch)

            if self.log_config.log_images_every[1] == "epochs" and epoch % self.log_config.log_images_every[0] == 0:
                log_images(engine, IN_TRAINING, epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(engine: Engine):
            if engine.state.epoch % self.log_config.validate_every_nth_epoch == 0:
                # evaluate on training data
                evaluator.mode = TRAINING
                evaluator.run(train_loader_eval)
                self.logger.info(
                    "Training Results  -  Epoch: %d  Avg loss: %.3f",
                    engine.state.epoch,
                    evaluator.state.metrics[LOSS_NAME],
                )
                log_eval(evaluator, TRAINING, engine.state.epoch)

                # evaluate on validation data
                evaluator.mode = VALIDATION
                evaluator.run(valid_loader)
                self.logger.info(
                    "Validation Results - Epoch: %d  Avg loss: %.3f",
                    engine.state.epoch,
                    evaluator.state.metrics[LOSS_NAME],
                )
                log_eval(evaluator, VALIDATION, engine.state.epoch)
                evaluator.fire_event(CustomEvents.VALIDATION_DONE)

        @trainer.on(Events.COMPLETED)
        def test(engine: Engine):
            evaluator.mode = TEST
            evaluator.run(test_loader)
            self.logger.info(
                "Test Results    -    Epoch: %d  Avg loss: %.3f", engine.state.epoch, evaluator.state.metrics[LOSS_NAME]
            )
            log_eval(evaluator, TEST, engine.state.epoch)

        trainer.run(train_loader, max_epochs=self.max_num_epochs)
        writer.close()


class SubsetSequentialSampler(torch.utils.data.sampler.Sampler):
    """Samples elements in fixed order from a given list of indices.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices: Sequence[int]):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ResultSaver:
    def __init__(self, *names: str, file_path: Path):
        self.folders = {name: file_path / name for name in names}
        for dir in self.folders.values():
            dir.mkdir(parents=True)

    def save(self, name: str, batch: torch.tensor, at: int):
        if name not in self.folders:
            return

        batch = torch.sigmoid(batch.detach()).cpu().numpy()
        assert len(batch.shape) == 3, batch.shape
        batch = (batch * numpy.iinfo(numpy.uint8).max).astype(numpy.uint8)
        for i, img in enumerate(batch, start=at):
            Image.fromarray(img).save(self.folders[name] / f"seg{i:04}.tif")
