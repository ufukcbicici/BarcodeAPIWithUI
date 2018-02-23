package bandrol_training.controller;

import bandrol_training.model.*;
import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Point2D;
import javafx.scene.Node;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.KeyEvent;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.Background;
import javafx.scene.layout.BackgroundFill;
import javafx.scene.layout.CornerRadii;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import javafx.scene.shape.Rectangle;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import javafx.stage.Window;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;

public class Controller {
    private Stage primaryStage;
    @FXML
    private Button open_image_btn;
    @FXML
    private Button start_labeling_img_btn;
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
    @FXML
    private Label sliding_window_x;
    @FXML
    private Label sliding_window_y;
    @FXML
    private Label curr_label_txt;
    @FXML
    private Button set_sliding_window_size_btn;
    @FXML
    private Button set_main_window_size_btn;
    @FXML
    private TextField reference_width_tf;
    @FXML
    private ComboBox<String> label_selection_cmbox;
    @FXML
    private TextField max_rotation_angle_tf;
    @FXML
    private TextField step_angle_tf;
    @FXML
    private TextField max_horizontal_offset_tf;
    @FXML
    private TextField horizontal_step_tf;
    @FXML
    private TextField max_vertical_offset_tf;
    @FXML
    private TextField vertical_step_tf;

    public Controller()
    {
        super();
        fileChooser = new FileChooser();
        listOfClickPoints = new ArrayList<>();
        listOfImagePoints = new ArrayList<>();
        LabelingStateContainer.reset();
    }

    public void initUIElements()
    {
        label_selection_cmbox.getItems().addAll(
                "A","B","C","D","E","F","G","H","I","J","K",
                "L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
                "0","1","2","3","4","5","6","7","8","9"
        );
        label_selection_cmbox.getSelectionModel().select("A");
    }

    @FXML
    public void modeChanged(ActionEvent action)
    {
        clearSceneInfo();
        System.out.println("pipeline_radio_btn:" + pipeline_radio_btn.isSelected());
        System.out.println("labeling_radio_btn:" + labeling_radio_btn.isSelected());
        if(labeling_radio_btn.isSelected())
        {
            startLabelImage();
        }
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
            Image img;
            try
            {
                img = new Image(currentFile.toURI().toString());
            }
            catch (Exception e)
            {
                System.err.println(e.toString());
                return;
            }
            primaryStage.setTitle(currentFile.getName());
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
            if(labeling_radio_btn.isSelected())
                startLabelImage();
//            Bounds bounds1 = bandrol_imageview.getBoundsInLocal();
//            Bounds bounds2 = central_pane.getBoundsInParent();
//            System.out.println("ImageView Width:"+bandrol_imageview.getBoundsInParent().getWidth());
//            System.out.println("ImageView Height:"+bandrol_imageview.getBoundsInParent().getHeight());
        }
    }

    @FXML
    private void startLabelImage()
    {
        // Reset labeling state variables
        LabelingStateContainer.reset();

        // Set up a new training image, init state variables
        BufferedImage awtImg = SwingFXUtils.fromFXImage(currImage, null);
        Mat sourceOpenCVImg = Utils.bufferedImageToMat(awtImg);
        Mat resizedSource = new Mat();
        double referenceWidth = Double.parseDouble(reference_width_tf.getText());
        double resizeRatio = referenceWidth / sourceOpenCVImg.cols();
        Imgproc.resize(sourceOpenCVImg, resizedSource,
                new Size(resizeRatio*sourceOpenCVImg.cols(),resizeRatio*sourceOpenCVImg.rows()));
        LabelingStateContainer.sourceTrainingImg = resizedSource;
        BufferedImage resizedBufImg = Utils.matToBufferedImage(resizedSource, null);
        assert resizedBufImg != null;
        currImage = SwingFXUtils.toFXImage(resizedBufImg, null);
        bandrol_imageview.setImage(null);
        bandrol_imageview.setFitWidth(resizedSource.cols());
        bandrol_imageview.setFitHeight(resizedSource.rows());
        bandrol_imageview.setImage(currImage);
        double sliding_window_width = Double.parseDouble(sliding_window_width_tf.getText());
        double sliding_window_height = Double.parseDouble(sliding_window_height_tf.getText());
        LabelingStateContainer.currBB =
                addNewSlidingWindowRectangle(0,0,sliding_window_width, sliding_window_height, Color.RED);


    }

    @FXML
    public void startAnnotation(ActionEvent actionEvent)
    {
        System.out.println(bandrol_imageview.getBoundsInParent());
        System.out.println(bandrol_imageview.getBoundsInLocal());
        // Step 1) Write all ground truth boxes into the db
        DbUtils.writeGroundTruth(new ArrayList<>(LabelingStateContainer.groundTruthMap.values()));
        // Step 2) Create ground truth positive samples with proper data augmentation
        double minRotationAngle = -Double.parseDouble(max_rotation_angle_tf.getText());
        double stepRotationAngle = Double.parseDouble(step_angle_tf.getText());
        double maxRotationAngle = -minRotationAngle;
        double minHorizontalOffset = -Double.parseDouble(max_horizontal_offset_tf.getText());
        double stepHorizontal = Double.parseDouble(horizontal_step_tf.getText());
        double maxHorizontalOffset = -minHorizontalOffset;
        double minVerticalOffset = -Double.parseDouble(max_vertical_offset_tf.getText());
        double stepVertical = Double.parseDouble(vertical_step_tf.getText());
        double maxVerticalOffset = -minVerticalOffset;
        for(Rectangle rect : LabelingStateContainer.groundTruthMap.keySet())
        {
            GroundTruth gt = LabelingStateContainer.groundTruthMap.get(rect);
            DataGenerator.augmentSample(currentFile.getName(),
                    LabelingStateContainer.sourceTrainingImg, gt,
                    minRotationAngle, stepRotationAngle, maxRotationAngle,
                    minHorizontalOffset, stepHorizontal, maxHorizontalOffset,
                    minVerticalOffset, stepVertical, maxVerticalOffset);
        }


        // Rotation angles
//        List<Double> rotationAngles = new ArrayList<>();
//        double minAngle = -15.0;
//        double stepAngle = 0.25;
//        double maxAngle = 15.0;
//        double currAngle = minAngle;
//        while (currAngle < maxAngle) {
//            rotationAngles.add(currAngle);
//            currAngle += stepAngle;
//        }
//        rotationAngles.toArray();
//        // Translation amounts
//        List<Double> verticalTranslation = new ArrayList<>();
//        List<Double> horizontalTranslation = new ArrayList<>();
//        double minOffset = -5.0;
//        double stepOffset = 1.0;
//        double maxOffset = 5.0;
//
//
//
//
//        for(Rectangle rect : LabelingStateContainer.groundTruthMap.keySet())
//        {
////            GroundTruth gt = LabelingStateContainer.groundTruthMap.get(rect);
////            DataGenerator.augmentSample(
////                    LabelingStateContainer.sourceTrainingImg, gt, );
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
            if(currImage == null)
                return;
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

    @FXML
    public void onAddGroundTruth(ActionEvent ae)
    {
        addGroundTruth();
    }

    @FXML
    public void onKeyReleased(KeyEvent ke)
    {
        if(!labeling_radio_btn.isSelected())
            return;
        if(LabelingStateContainer.currBB == null || LabelingStateContainer.sourceTrainingImg == null)
            return;
        double horizontalLimit = LabelingStateContainer.sourceTrainingImg.cols()
                - Double.parseDouble(sliding_window_width_tf.getText());
        double verticalLimit = LabelingStateContainer.sourceTrainingImg.rows()
                - Double.parseDouble(sliding_window_height_tf.getText());
        boolean didWindowChange = false;
        switch (ke.getText())
        {
            case "d":
                if (LabelingStateContainer.currBB.getX() < horizontalLimit) {
                    LabelingStateContainer.currBB.setX(LabelingStateContainer.currBB.getX() + 1.0);
                    didWindowChange = true;
                }
                break;
            case "a":
                if (LabelingStateContainer.currBB.getX() > 0) {
                    LabelingStateContainer.currBB.setX(LabelingStateContainer.currBB.getX() - 1.0);
                    didWindowChange = true;
                }
                break;
            case "w":
                if (LabelingStateContainer.currBB.getY() > 0) {
                    LabelingStateContainer.currBB.setY(LabelingStateContainer.currBB.getY() - 1.0);
                    didWindowChange = true;
                }
                break;
            case "s":
                if (LabelingStateContainer.currBB.getY() < verticalLimit) {
                    LabelingStateContainer.currBB.setY(LabelingStateContainer.currBB.getY() + 1.0);
                    didWindowChange = true;
                }
                break;
            case "l":
                addGroundTruth();
                break;
            case "z":
                if(LabelingStateContainer.rectangleList.size() > 0)
                {
                    int oldIndex = LabelingStateContainer.currentSelectedIndex;
                    LabelingStateContainer.currentSelectedIndex = LabelingStateContainer.getPrevIndex();
                    LabelingStateContainer.rectangleList.get(oldIndex).setStroke(Color.BLUE);
                    LabelingStateContainer.rectangleList.get(LabelingStateContainer.currentSelectedIndex).
                            setStroke(Color.ALICEBLUE);
                }
                break;
            case "x":
                if(LabelingStateContainer.rectangleList.size() > 0)
                {
                    int oldIndex = LabelingStateContainer.currentSelectedIndex;
                    LabelingStateContainer.currentSelectedIndex = LabelingStateContainer.getNextIndex();
                    LabelingStateContainer.rectangleList.get(oldIndex).setStroke(Color.BLUE);
                    LabelingStateContainer.rectangleList.get(LabelingStateContainer.currentSelectedIndex).
                            setStroke(Color.ALICEBLUE);

                }
                break;
        }
        switch (ke.getCode())
        {
            case DELETE:
                if(LabelingStateContainer.rectangleList.size() > 0)
                {
                    Rectangle toDelete = LabelingStateContainer.rectangleList.get(LabelingStateContainer.currentSelectedIndex);
                    LabelingStateContainer.groundTruthMap.remove(toDelete);
                    LabelingStateContainer.rectangleList.remove(toDelete);
                    central_pane.getChildren().remove(toDelete);
                    LabelingStateContainer.currentSelectedIndex = LabelingStateContainer.getPrevIndex();
                    if(LabelingStateContainer.rectangleList.size() > 0)
                        LabelingStateContainer.rectangleList.get(LabelingStateContainer.currentSelectedIndex).
                                setStroke(Color.ALICEBLUE);
                }
                break;
        }

        if(didWindowChange)
        {
            Mat slidingWindowContent = LabelingStateContainer.sourceTrainingImg.submat(
                    (int)LabelingStateContainer.currBB.getY(), (int)(LabelingStateContainer.currBB.getY() +
                            LabelingStateContainer.currBB.getHeight()),
                    (int)LabelingStateContainer.currBB.getX(),
                    (int)(LabelingStateContainer.currBB.getX() + LabelingStateContainer.currBB.getWidth()));
//            System.out.println(slidingWindowContent.rows());
//            System.out.println(slidingWindowContent.cols());
            sliding_window_large_image_view.setImage(
                    SwingFXUtils.toFXImage(
                            Objects.requireNonNull(Utils.matToBufferedImage(slidingWindowContent, null)),
                            null));
//            System.out.println(sliding_window_large_image_view.getBoundsInParent().getWidth());
//            System.out.println(sliding_window_large_image_view.getBoundsInParent().getHeight());
        }

    }

    private void addGroundTruth()
    {
        // Get ground truth label
        String currentLabel = label_selection_cmbox.getSelectionModel().getSelectedItem();
        LabelingStateContainer.currBB.setStroke(Color.BLUE);
        LabelingStateContainer.groundTruthMap.put(
                LabelingStateContainer.currBB, new GroundTruth(currentFile.getName(), currentLabel,
                        (int)LabelingStateContainer.currBB.getX(), (int)LabelingStateContainer.currBB.getY(),
                (int)LabelingStateContainer.currBB.getWidth(), (int)LabelingStateContainer.currBB.getHeight()));
        LabelingStateContainer.rectangleList.add(LabelingStateContainer.currBB);
        LabelingStateContainer.currBB =
                addNewSlidingWindowRectangle(LabelingStateContainer.currBB.getX(), LabelingStateContainer.currBB.getY(),
                        LabelingStateContainer.currBB.getWidth(), LabelingStateContainer.currBB.getHeight(),
                        Color.RED);
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

    private Rectangle addNewSlidingWindowRectangle(double x, double y, double width, double height, Color color)
    {
        Rectangle rect = new Rectangle(x,y,width,height);
        rect.setFill(null);
        rect.setStroke(color);
        rect.setStrokeWidth(1.0);
        central_pane.getChildren().add(rect);
        return rect;
    }

    private void clearSceneInfo()
    {
        // Label info
        LabelingStateContainer.reset();
        // Pipeline info
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

    public void setPrimaryStage(Stage primaryStage) {
        this.primaryStage = primaryStage;
    }
}

