package bandrol_training.model;

import org.opencv.core.Rect;

public class Detection
{
    private Rect rect;
    private Double response;

    public Detection(Rect rect, double response)
    {
        this.rect = rect;
        this.response = response;
    }

    public Rect getRect() {
        return rect;
    }

    public Double getResponse() {
        return response;
    }
}