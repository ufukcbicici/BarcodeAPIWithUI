package bandrol_training.post_processing;

import bandrol_training.model.Detection;

import java.util.ArrayList;
import java.util.List;

class CandidateSerial
{
    double probability;
    double logProbability;
    Detection topLeftDetection;
    List<Detection> detections;
    int numOfFalseDetections;

    public CandidateSerial(Detection topLeftDetection, double probability, double logProbability)
    {
        this.probability = 1.0;
        this.logProbability = 0.0;
        this.numOfFalseDetections = 0;
        this.topLeftDetection = topLeftDetection;
        detections = new ArrayList<>();
        addNewDetection(this.topLeftDetection, probability, logProbability);
    }

    public void addNewDetection(Detection detection, double probability, double logProbability)
    {
        detections.add(detection);
        this.probability *= probability;
        this.logProbability += logProbability;
        if(!detection.isActualDetection())
            numOfFalseDetections++;
    }

    public double getProbability()
    {
        return probability;
    }

    public double getLogProbability()
    {
        return logProbability;
    }

    public Detection getTopLeftDetection()
    {
        return topLeftDetection;
    }

    public List<Detection> getDetections()
    {
        return detections;
    }

    public int getNumOfFalseDetections()
    {
        return numOfFalseDetections;
    }
}
