package bandrol_training.post_processing;

import org.opencv.core.Mat;

class OLSResult
{
    private Mat weights;
    private double sumOfSquaredResiduals;
    private double variance;

    public OLSResult(Mat weights, double variance, double sumOfSquaredResiduals)
    {
        this.weights = weights;
        this.variance = variance;
        this.sumOfSquaredResiduals = sumOfSquaredResiduals;
    }

    public Mat getWeights() {
        return weights;
    }

    public double getVariance() {
        return variance;
    }

    public double getSumOfSquaredResiduals() {
        return sumOfSquaredResiduals;
    }
}
