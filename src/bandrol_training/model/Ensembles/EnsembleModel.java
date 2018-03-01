package bandrol_training.model.Ensembles;

import org.opencv.core.Mat;
import org.opencv.ml.StatModel;

import java.util.List;

public abstract class EnsembleModel {
    protected List<StatModel> models;

    public int getModelCount()
    {
        return models.size();
    }

    public void addModel(StatModel model)
    {
        models.add(model);
    }

    abstract public Mat predictByVoting(Mat samples);

    abstract public Mat predictByWeightedVoting(Mat samples) throws Exception;

    abstract public void loadEnsemble(int ensembleCount, String label);

    abstract public void saveEnsemble(String token);

    abstract public void trainSingleModel(Mat samples, Mat labels);

    abstract public Mat predictLabels(Mat samples);

    abstract public Mat predictConfidences(Mat samples);
}
