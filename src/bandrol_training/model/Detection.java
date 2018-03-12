package bandrol_training.model;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;

public class Detection
{
    private Rect rect;
    private Double response;
    private String label;
    private Mat topLeftVec;
    private boolean isActualDetection;

    public Detection(Rect rect, double response, String label)
    {
        this.rect = rect;
        this.response = response;
        this.label = label;
        this.topLeftVec = new Mat(3, 1, CvType.CV_64F);
        this.topLeftVec.put(0,0, rect.x);
        this.topLeftVec.put(1,0, rect.y);
        this.topLeftVec.put(2,0, 1.0);
        this.isActualDetection = true;
    }

    public Detection(Rect rect, double response, String label, boolean isActualDetection)
    {
        this.rect = rect;
        this.response = response;
        this.label = label;
        this.topLeftVec = new Mat(3, 1, CvType.CV_64F);
        this.topLeftVec.put(0,0, rect.x);
        this.topLeftVec.put(1,0, rect.y);
        this.topLeftVec.put(2,0, 1.0);
        this.isActualDetection = isActualDetection;
    }

    public Rect getRect() {
        return rect;
    }

    public Mat getTopLeftVec()
    {
        return topLeftVec;
    }

    public Double getResponse() {
        return response;
    }

    public String getLabel() {
        return label;
    }

    public boolean isActualDetection() {
        return isActualDetection;
    }

}