package bandrol_training.model.Algorithms;

import bandrol_training.model.Detection;
import bandrol_training.model.Utils;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class NonMaximaSuppression {

    public static List<Detection> run(List<Detection> preListOfDetections, double nms_iou_threshold)
    {
        List<Detection> sortedList = preListOfDetections.stream().
                sorted((d0,d1) -> -d0.getResponse().compareTo(d1.getResponse())).collect(Collectors.toList());
        List<Detection> maxima = new ArrayList<>();
        while (sortedList.size() > 0)
        {
            Detection mostConfidentDetection = sortedList.get(0);
            List<Detection> survivedList = new ArrayList<>();
            maxima.add(mostConfidentDetection);
            for(int i=1;i<sortedList.size();i++)
            {
                Detection candidate = sortedList.get(i);
                double iou = Utils.calculateIoU(mostConfidentDetection.getRect(), candidate.getRect());
                if(iou < nms_iou_threshold)
                    survivedList.add(candidate);
            }
            sortedList = survivedList;
        }
        return maxima;
    }
}
