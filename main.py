# main
if __name__ == "__main__":
    while True:
        model = input("Select model to run (dcgan1, dcgan2, wgan1d, wgan2d): ")
        try:
            if model == "dcgan1":
                import models.dcganone
                models.dcganone.train(0)
            elif model == "dcgan2":
                import models.dcgantwo
                models.dcgantwo.test()
            elif model == "wgan1d":
                import models.wganoned
                models.wganoned.train()
            elif model == "wgan2d":
                import models.wgantwod
                models.wgantwod.test()
            else:
                raise ValueError
        except ValueError:
            print("")
        else:
            break
