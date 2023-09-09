from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec 

class TrainingTracker(CallbackAny2Vec):
    def __init__(self, model_loss, total_epochs):
        self.epoch = 0
        self.loss_to_be_subed = 0
        self.model_loss = model_loss
        self.total_epochs = total_epochs

    def on_epoch_end(self, model):
        self.epoch += 1
        percentage = (self.epoch / self.total_epochs) * 100
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        self.model_loss.append(loss_now)

        if percentage % 1 == 0:
            print("Epoch {}/{} completed, --> {}% with training loss: {:.4f}".
                  format(self.epoch, self.total_epochs, int(percentage), loss_now))
            
class Training:
     @staticmethod
     def create_model(sentences):
        total_epochs = 10
        model_loss = []
        tracker = TrainingTracker(model_loss, total_epochs)
        model = Word2Vec(
            sentences,
            workers=8,
            sg=1,
            hs=0,
            window=len(max(sentences, key=len)),
            epochs=total_epochs,
            negative=10,
            seed=42,
            alpha=0.05,
            compute_loss=True,
            callbacks=[tracker]
        )
        return model

