from simpletransformers.classification import ClassificationModel, ClassificationArgs

def load_model(model_path, use_cuda=False, verbose=True):
    model = ClassificationModel('xlmroberta', model_path, num_labels=1, use_cuda=False)
    model = model.model.from_pretrained(model_path+"/pytorch_model.bin")
    if verbose:
        print("Intimacy model has been loaded correctly")
    return model

def predict_intimacy(model, text):    
    predictions, _ = model.predict(text)
    return predictions