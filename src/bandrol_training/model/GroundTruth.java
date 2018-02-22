package bandrol_training.model;

public class GroundTruth implements java.io.Serializable
{
    public String fileName;
    public String label;
    public int x;
    public int y;
    public int width;
    public int height;

    public GroundTruth(String fName, String l, int _x, int _y, int w, int h)
    {
        fileName = fName;
        label = l;
        x = _x;
        y = _y;
        width = w;
        height = h;
    }
}
