from summer.datasets import DIC_C2DH_HeLa, Fluo_C2DL_MSC, Fluo_N2DH_GOWT1, Fluo_N2DL_HeLa, PhC_C2DH_U373, PhC_C2DL_PSC, Fluo_N2DH_SIM


def run():
    # for D in [DIC_C2DH_HeLa, Fluo_C2DL_MSC, Fluo_N2DH_GOWT1, Fluo_N2DL_HeLa, PhC_C2DH_U373, PhC_C2DL_PSC, Fluo_N2DH_SIM]:
    #     D()

    ds = DIC_C2DH_HeLa(two=True)
    raw, label = ds[500]
    print(raw.size, label.size)

    print("done")


