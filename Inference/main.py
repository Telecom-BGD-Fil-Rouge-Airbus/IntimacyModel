from inference_intimacy import load_model, predict_intimacy

if __name__ == '__main__':
    clear_examples = ["You mean the world to me.", "Can you pass me the salt, please?", "I feel so safe and loved when I'm in your arms.", "What time is our meeting tomorrow?", "You know me better than anyone else.", "Do you have any plans for the weekend?", "I cherish every moment we spend together.", "I need to finish this report by the end of the day.", "I can't imagine my life without you.", "Could you remind me to buy groceries later?"]

    model_path = "model_intimacy"
    model = load_model(model_path, use_cuda=False, verbose=True)

    prediction = predict_intimacy(model, clear_examples)
    for i in range(len(prediction)):
        print("text :",clear_examples[i])
        print("prediction :", prediction[i])