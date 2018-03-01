package bandrol_training.model;

import org.opencv.core.Rect;

public class Detection
{
    private Rect rect;
    private Double response;
    private String label;

    public Detection(Rect rect, double response, String label)
    {
        this.rect = rect;
        this.response = response;
        this.label = label;
    }

    public Rect getRect() {
        return rect;
    }

    public Double getResponse() {
        return response;
    }

    public String getLabel() {
        return label;
    }
}