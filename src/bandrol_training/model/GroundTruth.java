package bandrol_training.model;

import org.opencv.core.Mat;
import org.opencv.core.Rect;

import java.nio.ByteBuffer;

public class GroundTruth implements java.io.Serializable
{
    public String fileName;
    public String label;
    public int x;
    public int y;
    public int width;
    public int height;
    public double iouWithClosestGroundTruth;
    private Mat hogFeature;

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

    public Mat getHogFeature() {
        return hogFeature;
    }

    public void setHogFeature(Mat hogFeature) {
        this.hogFeature = hogFeature;
    }

    public byte[] getHogFeatureAsByteArr()
    {
        ByteBuffer bb = ByteBuffer.allocate(hogFeature.cols() * hogFeature.rows() * 8);
        for(int i=0;i<hogFeature.rows();i++)
            bb.putDouble(hogFeature.get(i,0)[0]);
        byte [] bArr = bb.array();
        return bArr;
    }

    public String toString()
    {
        return "File:"+fileName+" "+"Label:"+label+" ("+x+","+y+") "+ " ("+width+","+height+")";
    }
}
