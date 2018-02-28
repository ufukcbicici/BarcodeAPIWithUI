//package bandrol_training.model;
//
//import java.util.*;
//
//public class EmpiricalDistribution
//{
//    private Map<Integer, List<Double>> cdfContainer;
//    private double histogramBinLength;
//
//    public EmpiricalDistribution(double histogramBinLength)
//    {
//        this.histogramBinLength = histogramBinLength;
//        cdfContainer = new HashMap<>();
//    }
//
//    public void insertSample(double x)
//    {
//        int lowerBound = (int)Math.floor(x / histogramBinLength);
//        if(!cdfContainer.containsKey(lowerBound))
//            cdfContainer.put(lowerBound, new ArrayList<>());
//        List<Double> binEntries = cdfContainer.get(lowerBound);
//        binEntries.add()
//        cdfContainer.put(lowerBound, frequency+1);
//    }
//
//    public double getProbabilityLessEqual(double x)
//    {
//        int sampleLowerBound = (int)Math.floor(x / histogramBinLength);
//        Set<Integer> binLowerBounds = cdfContainer.keySet();
//        for(Integer binLowerBound : binLowerBounds)
//        {
//            if(binLowerBound < sampleLowerBound)
//            {
//
//            }
//        }
//    }
//}
