package bandrol_training.controller;

import javafx.beans.value.ChangeListener;
import javafx.beans.value.ObservableValue;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyEvent;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.stage.FileChooser;
import javafx.stage.Window;
import javafx.scene.input.MouseEvent;
import javafx.stage.WindowEvent;
import org.opencv.core.Point;
import bandrol_training.model.Algorithm;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Controller {
    @FXML
    private Button open_image_btn;
    @FXML
    private Button detect_qr_code_btn;
    @FXML
    private Button execute_pipeline_btn;
    @FXML
    private Pane central_pane;
    @FXML
    private ImageView bandrol_imageview;
    private Image currImage;
    private File currentFile;
    private List<Point2D> listOfClickPoints;
    private List<Point2D> listOfImagePoints;
    private FileChooser fileChooser;
    private boolean isInPipelineMode;
    @FXML
    private RadioButton pipeline_radio_btn;
    @FXML
    private RadioButton labeling_radio_btn;
    @FXML
    private ToggleGroup radioGroup;
    @FXML
    private TextField sliding_window_width_tf;
    @FXML
    private TextField sliding_window_height_tf;
    @FXML
    private Label image_name_label;
    @FXML
    private Label image_height_lbl;
    @FXML
    private Label image_width_lbl;
    @FXML
    private ImageView sliding_window_large_image_view;
    @FXML
    private TextField negative_iou_tf;

    public Controller()
    {
        super();
        fileChooser = new FileChooser();
        listOfClickPoints = new ArrayList<>();
        listOfImagePoints = new ArrayList<>();
        isInPipelineMode = true;
    }

    @FXML
    public void modeChanged(ActionEvent action)
    {
        clearSceneInfo();
        System.out.println("pipeline_radio_btn:" + pipeline_radio_btn.isSelected());
        System.out.println("labeling_radio_btn:" + labeling_radio_btn.isSelected());
    }

    @FXML
    public void openImage(ActionEvent actionEvent)
    {
        clearSceneInfo();
        Node source = (Node) actionEvent.getSource();
        Window stage = source.getScene().getWindow();
        currentFile = fileChooser.showOpenDialog(stage);
        if(currentFile != null)
        {
            Image img = null;
            try
            {
                img = new Image(currentFile.toURI().toString());
            }
            catch (Exception e)
            {
                System.err.println(e.toString());
                return;
            }
            if(img.getWidth() <= 0.0 && img.getHeight() <= 0.0) {
                System.out.println("Not a correct image.");
                return;
            }
            central_pane.setBackground(new Background(new BackgroundFill(Color.web("#" + "ffffff"),
                    CornerRadii.EMPTY, Insets.EMPTY)));
            bandrol_imageview.setImage(img);
            currImage = img;
            System.out.println("Image Width:"+img.getWidth());
            System.out.println("Image Height:"+img.getHeight());
//            Bounds bounds1 = bandrol_imageview.getBoundsInLocal();
//            Bounds bounds2 = central_pane.getBoundsInParent();
//            System.out.println("ImageView Width:"+bandrol_imageview.getBoundsInParent().getWidth());
//            System.out.println("ImageView Height:"+bandrol_imageview.getBoundsInParent().getHeight());
        }
    }

    @FXML
    public void labelImage()
    {

    }

    @FXML
    public void getQRCode(ActionEvent actionEvent)
    {
//        // addNewGuidanceLine(new Point2D(0, 0), new Point2D(200, 200));
//        if (listOfImagePoints.size() == 4)
//        {
//            BufferedImage awtImg = SwingFXUtils.fromFXImage(currImage, null);
//            List<Point> opencvPointList = new ArrayList<>();
//            for(Point2D fxPoint : listOfImagePoints)
//            {
//                Point opencvPoint = new Point(fxPoint.getX(), fxPoint.getY());
//                opencvPointList.add(opencvPoint);
//                System.out.println("(x:"+opencvPoint.x+","+opencvPoint.y+")");
//            }
//            List<QRCodePoint> qrCodePointList = Algorithm.getQRCodePoints(awtImg, opencvPointList);
//            if(qrCodePointList == null)
//                return;
//            // Draw QrCodePoints
//            for(QRCodePoint qrCodePoint : qrCodePointList)
//            {
//                Point2D imageViewPoint = convertFromImageToImageViewCoords(new Point2D(qrCodePoint.getLocation().x,
//                        qrCodePoint.getLocation().y));
//                addNewCircle(imageViewPoint.getX(), imageViewPoint.getY(), 5.0, Color.BLUE);
//            }
//            // Draw Up Vector
//            PipelineInfo pipelineInfo = Algorithm.getOrientationFromQRCodePoints(awtImg, qrCodePointList);
//            Mat upVector = pipelineInfo.getRotationUpVectorStartFinishPoints();
//            Point2D up0 = convertFromImageToImageViewCoords(
//                    new Point2D(upVector.get(0,0)[0], upVector.get(0,1)[0]));
//            Point2D up1 = convertFromImageToImageViewCoords(
//                    new Point2D(upVector.get(1,0)[0], upVector.get(1,1)[0]));
//            addNewGuidanceLine(up0, up1, Color.YELLOWGREEN);
//        }
    }

    @FXML
    public void executePipeline(ActionEvent actionEvent)
    {
        if (listOfImagePoints.size() == 4)
        {
            BufferedImage awtImg = SwingFXUtils.fromFXImage(currImage, null);
            List<Point> opencvPointList = new ArrayList<>();
            for(Point2D fxPoint : listOfImagePoints)
            {
                Point opencvPoint = new Point(fxPoint.getX(), fxPoint.getY());
                opencvPointList.add(opencvPoint);
                System.out.println("(x:"+opencvPoint.x+","+opencvPoint.y+")");
            }
            Algorithm.execute(awtImg, opencvPointList);
        }
    }

    @FXML
    public void onMouseClickedOverImage(MouseEvent me)
    {
        if(!labeling_radio_btn.isSelected() && pipeline_radio_btn.isSelected())
        {
            if(listOfClickPoints.size() == 4)
            {
                clearSceneInfo();
            }
            //Parent parent = bandrol_imageview.getParent();
            Point2D localPoint = new Point2D(me.getX(), me.getY());
            Point2D transformedScreenPoint = bandrol_imageview.localToParent(localPoint);
            Point2D transformedImagePoint = convertFromImageViewToImageCoords(localPoint);
            listOfClickPoints.add(transformedScreenPoint);
            listOfImagePoints.add(transformedImagePoint);
            System.out.println("X:"+me.getX()+" Y:"+me.getY());
            // Draw a new line
            if(listOfClickPoints.size() > 1)
            {
                Point2D p0 = listOfClickPoints.get(listOfClickPoints.size()-2);
                Point2D p1 = listOfClickPoints.get(listOfClickPoints.size()-1);
                addNewGuidanceLine(p0,p1,Color.RED);
            }
            if(listOfClickPoints.size() == 4)
            {
                Point2D p0 = listOfClickPoints.get(listOfClickPoints.size()-1);
                Point2D p1 = listOfClickPoints.get(0);
                addNewGuidanceLine(p0,p1,Color.RED);
            }
        }

    }

    @FXML void onKeyReleased(KeyEvent ke)
    {

    }

    private void addNewCircle(double centerx, double centery, double radius, Color color)
    {
        Circle circle = new Circle(centerx, centery, radius, color);
        central_pane.getChildren().add(circle);
    }

    private void addNewGuidanceLine(Point2D p0, Point2D p1, Color color)
    {
        Line line = new Line(p0.getX(), p0.getY(), p1.getX(), p1.getY());
        //Line line = new Line(0, 0, 100, 100);
        line.setFill(null);
        line.setStroke(color);
        line.setStrokeWidth(2);
        central_pane.getChildren().add(line);
    }

    private void clearSceneInfo()
    {
        List<Node> nodeList = new ArrayList<>(central_pane.getChildren());
        for(Node n : nodeList)
        {
            if(n != bandrol_imageview)
                central_pane.getChildren().remove(n);
        }
        listOfClickPoints = new ArrayList<>();
        listOfImagePoints = new ArrayList<>();
    }

    private Point2D convertFromImageViewToImageCoords(Point2D imageViewPoint)
    {
        return new Point2D(
                (imageViewPoint.getX()/bandrol_imageview.getBoundsInParent().getWidth())*currImage.getWidth(),
                (imageViewPoint.getY()/bandrol_imageview.getBoundsInParent().getHeight())*currImage.getHeight());
    }

    private Point2D convertFromImageToImageViewCoords(Point2D imagePoint)
    {
        return new Point2D(
                (imagePoint.getX()/currImage.getWidth())*bandrol_imageview.getBoundsInParent().getWidth(),
                (imagePoint.getY()/currImage.getHeight())*bandrol_imageview.getBoundsInParent().getHeight());
    }
}

