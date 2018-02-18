package sample.controller;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.scene.Node;
import javafx.scene.control.Button;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.*;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.stage.FileChooser;
import javafx.stage.Window;
import javafx.scene.input.MouseEvent;
import org.opencv.core.Point;
import sample.model.Algorithm;
import sample.model.OrientationFinding.OrientationFinder;
import sample.model.QRCodeReading.QRCodePoint;

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
    private Pane central_pane;
    @FXML
    private ImageView bandrol_imageview;
    private Image currImage;
    private File currentFile;
    private List<Point2D> listOfClickPoints;
    private List<Point2D> listOfImagePoints;
    private List<Circle> listOfQRCodePoints;
    private List<Line> listOfSelectionLines;
    private FileChooser fileChooser;

    public Controller()
    {
        super();
        fileChooser = new FileChooser();
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
    public void getQRCode(ActionEvent actionEvent)
    {
        // addNewGuidanceLine(new Point2D(0, 0), new Point2D(200, 200));
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
            List<QRCodePoint> qrCodePointList = Algorithm.getQRCodePoints(awtImg, opencvPointList);
            if(qrCodePointList == null)
                return;
            // Draw QrCodePoints
            listOfQRCodePoints = new ArrayList<>();
            for(QRCodePoint qrCodePoint : qrCodePointList)
            {
                Point2D imageViewPoint = convertFromImageToImageViewCoords(new Point2D(qrCodePoint.getLocation().x,
                        qrCodePoint.getLocation().y));
                addNewCircle(imageViewPoint.getX(), imageViewPoint.getY(), 5.0, Color.BLUE);
            }
            Algorithm.getOrientationFromQRCodePoints(qrCodePointList);
        }


    }

    @FXML
    public void onMouseClickedOverImage(MouseEvent me)
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
            addNewGuidanceLine(p0,p1);
        }
        if(listOfClickPoints.size() == 4)
        {
            Point2D p0 = listOfClickPoints.get(listOfClickPoints.size()-1);
            Point2D p1 = listOfClickPoints.get(0);
            addNewGuidanceLine(p0,p1);
        }
    }

    private void addNewCircle(double centerx, double centery, double radius, Color color)
    {
        Circle circle = new Circle(centerx, centery, radius, color);
        listOfQRCodePoints.add(circle);
        central_pane.getChildren().add(circle);

    }

    private void addNewGuidanceLine(Point2D p0, Point2D p1)
    {
        Line line = new Line(p0.getX(), p0.getY(), p1.getX(), p1.getY());
        //Line line = new Line(0, 0, 100, 100);
        line.setFill(null);
        line.setStroke(Color.RED);
        line.setStrokeWidth(2);
        listOfSelectionLines.add(line);
        central_pane.getChildren().add(line);
    }

    private void clearSceneInfo()
    {
        listOfClickPoints = new ArrayList<>();
        listOfImagePoints = new ArrayList<>();
        if (listOfSelectionLines != null)
        {
            for(Line line : listOfSelectionLines)
                central_pane.getChildren().remove(line);
        }
        if(listOfQRCodePoints != null)
        {
            for(Circle circle : listOfQRCodePoints)
            {
                central_pane.getChildren().remove(circle);
            }
        }
        listOfSelectionLines = new ArrayList<>();
        listOfQRCodePoints = new ArrayList<>();
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

