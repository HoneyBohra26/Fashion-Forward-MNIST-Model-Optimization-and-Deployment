from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_training_and_validation import Training
from cnnClassifier import logger


STAGE_NAME = "Model Training and Validation"


class Training_and_Validation:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_val_split()
        training.train_model()
        training.save_model()


    
if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = Training_and_Validation()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e