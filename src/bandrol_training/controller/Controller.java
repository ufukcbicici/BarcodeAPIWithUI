package bandrol_training.controller;

import bandrol_training.Constants;
import bandrol_training.model.*;
import bandrol_training.model.Algorithm;
// import bandrol_training.model.Detectors.Detector1;
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
import jdk.jshell.spi.ExecutionControl;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

import static bandrol_training.model.LabelingStateContainer.currBB;

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
    @FXML
    private Label sliding_window_x_lbl;
    @FXML
    private Label sliding_window_y_lbl;
    @FXML
    private Label iou_lbl;
    @FXML
    private Label serial_num_label;
    @FXML
    private Button train_object_detectors_btn;
    @FXML
    private TextField max_iou_txt_fld;
    @FXML
    private Button detect_chars_btn;
    @FXML
    private TextField nms_iou_threshold_txt_fld;
    @FXML
    private TextField object_sign_txt_field;
    @FXML
    private Button char_classifier_btn;
    @FXML
    private Button detect_on_all_test;
    @FXML
    private Button method_0_btn;
    @FXML
    private Button method_1_btn;

    // private Detector1 detector1;


    private int objectDetectorEnsembleCount = 1;
    private int charClassifierEnsembleCount = 21;
    private String charToProcess = "0";

    public Controller()
    {
        super();
        fileChooser = new FileChooser();
        listOfClickPoints = new ArrayList<>();
        listOfImagePoints = new ArrayList<>();
        LabelingStateContainer.reset();
        // detector1 = null;
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
    public void run_method_0(ActionEvent actionEvent)
    {

    }

    @FXML
    public void run_method_1(ActionEvent actionEvent)
    {
//        if(detector1 == null)
//        {
//            detector1  = new Detector1(ClassifierType.SVM);
//            try
//            {
//                detector1.init(objectDetectorEnsembleCount);
//            } catch (ExecutionControl.NotImplementedException e) {
//                e.printStackTrace();
//            }
//        }

        // detector1.detect();

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
    public void onTestAllImages(ActionEvent actionEvent)
    {
        List<String> testImages = Utils.getAllTestImageNames();
        double sliding_window_width = Double.parseDouble(sliding_window_width_tf.getText());
        double sliding_window_height = Double.parseDouble(sliding_window_height_tf.getText());
        for(String fileName : testImages)
        {
            String path = Constants.TEST_IMAGES + fileName;
            Mat image = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            Mat resizedSource = new Mat();
            double referenceWidth = Double.parseDouble(reference_width_tf.getText());
            double resizeRatio = referenceWidth / image.cols();
            Imgproc.resize(image, resizedSource,
                    new Size(resizeRatio*image.cols(),resizeRatio*image.rows()));

            ObjectDetector.detectWithEnsembles(
                    objectDetectorEnsembleCount,
                    charToProcess,
                    resizedSource,
                    (int)sliding_window_width,
                    (int)sliding_window_height,
                    Double.parseDouble(nms_iou_threshold_txt_fld.getText()),
                    Double.parseDouble(object_sign_txt_field.getText()),
                    Double.parseDouble(reference_width_tf.getText()));
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
        currBB =
                addNewSlidingWindowRectangle(0,0,sliding_window_width, sliding_window_height, Color.RED);



    }

    @FXML
    public void detectCharacters(ActionEvent actionEvent)
    {
        double sliding_window_width = Double.parseDouble(sliding_window_width_tf.getText());
        double sliding_window_height = Double.parseDouble(sliding_window_height_tf.getText());

//        ObjectDetector.detectObjects(
//                LabelingStateContainer.sourceTrainingImg,
//                (int)sliding_window_width,
//                (int)sliding_window_height,
//                Double.parseDouble(nms_iou_threshold_txt_fld.getText()),
//                Double.parseDouble(object_sign_txt_field.getText()),
//                Double.parseDouble(reference_width_tf.getText()));

        ObjectDetector.detectWithEnsembles(objectDetectorEnsembleCount,
                charToProcess, LabelingStateContainer.sourceTrainingImg,
                (int)sliding_window_width,(int)sliding_window_height,
                Double.parseDouble(nms_iou_threshold_txt_fld.getText()),
                Double.parseDouble(object_sign_txt_field.getText()),
                Double.parseDouble(reference_width_tf.getText()));
    }

    @FXML
    public void startAnnotation(ActionEvent actionEvent)
    {
        System.out.println(bandrol_imageview.getBoundsInParent());
        System.out.println(bandrol_imageview.getBoundsInLocal());
        double sliding_window_width = Double.parseDouble(sliding_window_width_tf.getText());
        double sliding_window_height = Double.parseDouble(sliding_window_height_tf.getText());
        // Step 1) Get data augmentation parameters
        double minRotationAngle = -Double.parseDouble(max_rotation_angle_tf.getText());
        double stepRotationAngle = Double.parseDouble(step_angle_tf.getText());
        double maxRotationAngle = -minRotationAngle;
        double minHorizontalOffset = -Double.parseDouble(max_horizontal_offset_tf.getText());
        double stepHorizontal = Double.parseDouble(horizontal_step_tf.getText());
        double maxHorizontalOffset = -minHorizontalOffset;
        double minVerticalOffset = -Double.parseDouble(max_vertical_offset_tf.getText());
        double stepVertical = Double.parseDouble(vertical_step_tf.getText());
        double maxVerticalOffset = -minVerticalOffset;
        // Step 2) Create positive samples with data augmentation and extract HOG features for each augmented sample.
        List<GroundTruth> augmentedGroundTruths = new ArrayList<>();
        for(Rectangle rect : LabelingStateContainer.groundTruthMap.keySet())
        {
            GroundTruth gt = LabelingStateContainer.groundTruthMap.get(rect);
            System.out.println("Augmenting "+gt.toString());
            augmentedGroundTruths.addAll(DataGenerator.augmentSample(
                    currentFile.getName(),
                    LabelingStateContainer.sourceTrainingImg, gt,
                    minRotationAngle, stepRotationAngle, maxRotationAngle,
                    minHorizontalOffset, stepHorizontal, maxHorizontalOffset,
                    minVerticalOffset, stepVertical, maxVerticalOffset));
            System.out.println("Extracting HOG Features for "+gt.toString());
            for(GroundTruth augmentedGt : augmentedGroundTruths)
            {
                try
                {
                    Mat hogFeature = HOGExtractor.extractOpenCVHogFeature(augmentedGt.getImg(),
                            (int)sliding_window_width, (int)sliding_window_height);
                    // Mat hogFeatureOpenCV = HOGExtractor.extractOpenCVHogFeature(augmentedGt.getImg());
                    augmentedGt.setHogFeature(hogFeature);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }
        System.out.println("There are "+augmentedGroundTruths.size()+" ground truths.");
        System.out.println("Writing to DB.");
        DbUtils.writeGroundTruth(augmentedGroundTruths);
        // Step 3) Negative mining: Traverse all potential windows on the image, measure their IoUs with the closest
        // ground truth and record them as potential negative samples. We are going to use the samples with an IoU against
        // the nearest ground truth objects as negative samples for our object detector then.
        List<GroundTruth> negativeList = new ArrayList<>();
        for(int i=0;i<LabelingStateContainer.sourceTrainingImg.rows();i++)
        {
            for(int j=0;j<LabelingStateContainer.sourceTrainingImg.cols();j++)
            {
                if(i + (int)sliding_window_height - 1 >= LabelingStateContainer.sourceTrainingImg.rows())
                    continue;
                if(j + (int)sliding_window_width - 1 >= LabelingStateContainer.sourceTrainingImg.cols())
                    continue;
                Rect negativeSampleRect = new Rect(j, i, (int)sliding_window_width, (int)sliding_window_height);
                double max_iou = Utils.getMaxIoU(negativeSampleRect, LabelingStateContainer.groundTruthMap.values().stream().
                                map(GroundTruth::getBoundingRect).collect(Collectors.toList()));
                GroundTruth negativeSample = new GroundTruth(currentFile.getName(), "-1", j, i,
                        (int)sliding_window_width, (int)sliding_window_height, max_iou);
                Mat imgRect = LabelingStateContainer.sourceTrainingImg.submat(
                                        i, i + (int)sliding_window_height,
                                        j, j + (int)sliding_window_width);
                try
                {
                    Mat hogFeature = HOGExtractor.extractOpenCVHogFeature(imgRect,
                            (int)sliding_window_width, (int)sliding_window_height);
                    negativeSample.setHogFeature(hogFeature);
                    negativeList.add(negativeSample);
                } catch (Exception e)
                {
                    e.printStackTrace();
                }

            }
        }
        System.out.println("There are "+negativeList.size()+" negative samples.");
        System.out.println("Writing to DB.");
        DbUtils.writeGroundTruth(negativeList);
        System.out.println("Successfully Completed.");
    }

    @FXML
    public void onCharClassifier(ActionEvent actionEvent)
    {
        CharClassifier.train(charClassifierEnsembleCount, 0.01,0, 0.0);
    }

    @FXML
    public void onTrainObjectDetectors(ActionEvent actionEvent)
    {
        double sliding_window_width = Double.parseDouble(sliding_window_width_tf.getText());
        double sliding_window_height = Double.parseDouble(sliding_window_height_tf.getText());
        double img_width = Double.parseDouble(reference_width_tf.getText());
        double max_iou = Double.parseDouble(max_iou_txt_fld.getText());
        ObjectDetector.trainEnsemble(objectDetectorEnsembleCount, max_iou, img_width, charToProcess);
        //ObjectDetector.train(max_iou, img_width);
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
        if(currBB == null || LabelingStateContainer.sourceTrainingImg == null)
            return;
        double horizontalLimit = LabelingStateContainer.sourceTrainingImg.cols()
                - Double.parseDouble(sliding_window_width_tf.getText());
        double verticalLimit = LabelingStateContainer.sourceTrainingImg.rows()
                - Double.parseDouble(sliding_window_height_tf.getText());
        boolean didWindowChange = false;
        switch (ke.getText())
        {
            case "d":
                if (currBB.getX() < horizontalLimit) {
                    currBB.setX(currBB.getX() + 1.0);
                    didWindowChange = true;
                }
                break;
            case "a":
                if (currBB.getX() > 0) {
                    currBB.setX(currBB.getX() - 1.0);
                    didWindowChange = true;
                }
                break;
            case "w":
                if (currBB.getY() > 0) {
                    currBB.setY(currBB.getY() - 1.0);
                    didWindowChange = true;
                }
                break;
            case "s":
                if (currBB.getY() < verticalLimit) {
                    currBB.setY(currBB.getY() + 1.0);
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
                    (int) currBB.getY(), (int)(currBB.getY() +
                            currBB.getHeight()),
                    (int) currBB.getX(),
                    (int)(currBB.getX() + currBB.getWidth()));
            sliding_window_large_image_view.setImage(
                    SwingFXUtils.toFXImage(
                            Objects.requireNonNull(Utils.matToBufferedImage(slidingWindowContent, null)),
                            null));
            // Set UI Information
            sliding_window_x_lbl.setText("" + currBB.getX());
            sliding_window_y_lbl.setText("" + currBB.getY());
            double max_iou = Utils.getMaxIoU(
            new Rect((int)currBB.getX(), (int)currBB.getY(), (int)currBB.getWidth(), (int)currBB.getHeight()),
            LabelingStateContainer.groundTruthMap.values().stream().
                    map(GroundTruth::getBoundingRect).collect(Collectors.toList()));
            iou_lbl.setText("" + max_iou);


        }

    }

    private void addGroundTruth()
    {
        // Get ground truth label
        String currentLabel = label_selection_cmbox.getSelectionModel().getSelectedItem();
        serial_num_label.setText(serial_num_label.getText()+currentLabel);
        currBB.setStroke(Color.BLUE);
        LabelingStateContainer.groundTruthMap.put(
                currBB, new GroundTruth(currentFile.getName(), currentLabel,
                        (int) currBB.getX(), (int) currBB.getY(),
                (int) currBB.getWidth(), (int) currBB.getHeight()));
        LabelingStateContainer.rectangleList.add(currBB);
        currBB =
                addNewSlidingWindowRectangle(currBB.getX(), currBB.getY(),
                        currBB.getWidth(), currBB.getHeight(),
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

    @FXML
    private void onShowTrainingSamples()
    {
        List<String> trainingImages = Utils.getAllTrainingImageNames();
        String exlusionStatement = "FileName NOT IN" + Utils.getFileSelectionClause();
        String excludeClause = Utils.getFilterClause("Label != -1", exlusionStatement);
        List<GroundTruth> allTrainingSamples = DbUtils.readGroundTruths(excludeClause);
        for(String fileName : trainingImages)
        {
            List<GroundTruth> imageSamples = allTrainingSamples.stream().
                    filter(gt ->
                            gt.fileName.equals(fileName) &&
                            Math.abs(gt.rotation) < 0.0001 &&
                            Math.abs(gt.verticalDisplacement) < 0.0001 &&
                            Math.abs(gt.horizontalDisplacement) < 0.0001).collect(Collectors.toList());
            assert imageSamples.size() == 14;
            String path = Constants.TRAINING_IMAGES + fileName;
            Mat image = Imgcodecs.imread(path, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            Mat resizedSource = new Mat();
            double referenceWidth = Double.parseDouble(reference_width_tf.getText());
            double resizeRatio = referenceWidth / image.cols();
            Imgproc.resize(image, resizedSource,
                    new Size(resizeRatio*image.cols(),resizeRatio*image.rows()));
            for(GroundTruth gt : imageSamples)
            {
                Rect r = new Rect(gt.x, gt.y, gt.width, gt.height);
                String label = gt.label;
                int font = Core.FONT_HERSHEY_COMPLEX;
                Imgproc.rectangle(resizedSource, new Point(r.x, r.y),
                        new Point(r.x + r.width - 1, r.y + r.height - 1),
                        new Scalar(0, 0, 255));
                Imgproc.putText(resizedSource, label, new Point(r.x, r.y+10), font,
                        0.4,new Scalar(0,255,0),1);
            }
            Utils.showImageInPopup(Utils.matToBufferedImage(resizedSource, null));
        }
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
        serial_num_label.setText("");
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

