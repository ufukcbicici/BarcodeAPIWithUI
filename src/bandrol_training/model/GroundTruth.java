package bandrol_training.model;

public class GroundTruth implements java.io.Serializable
{
    public String label;
    public int x;
    public int y;
    public int width;
    public int height;

    public GroundTruth(String l, int _x, int _y, int w, int h)
    {
        label = l;
        x = _x;
        y = _y;
        width = w;
        height = h;
    }
}
