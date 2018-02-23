package bandrol_training.model;

import org.opencv.core.Rect;

public class GroundTruth implements java.io.Serializable
{
    public String fileName;
    public String label;
    public int x;
    public int y;
    public int width;
    public int height;
    public double iouWithClosestGroundTruth;

    public GroundTruth(String fName, String l, int _x, int _y, int w, int h)
    {
        fileName = fName;
        label = l;
        x = _x;
        y = _y;
        width = w;
        height = h;
        iouWithClosestGroundTruth = 1.0;
    }

    public GroundTruth(String fName, String l, int _x, int _y, int w, int h, double iou)
    {
        fileName = fName;
        label = l;
        x = _x;
        y = _y;
        width = w;
        height = h;
        iouWithClosestGroundTruth = iou;
    }

    public Rect getBoundingRect()
    {
        return new Rect(x, y, width, height);
    }
}
