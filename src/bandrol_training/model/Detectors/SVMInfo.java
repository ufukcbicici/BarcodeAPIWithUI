package bandrol_training.model.Detectors;

public class SVMInfo {
    private String fileName;
    private double positiveSign;
    private String label;

    public SVMInfo(String fileName, double positiveSign, String label)
    {
        this.fileName = fileName;
        this.positiveSign = positiveSign;
        this.label = label;
    }

    public String getFileName() {
        return fileName;
    }

    public double getPositiveSign() {
        return positiveSign;
    }

    public String getLabel() {
        return label;
    }
}
