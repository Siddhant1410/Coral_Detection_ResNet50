def main():
    import cv2
    from color_recognition_api import color_histogram_feature_extraction
    from color_recognition_api import knn_classifier
    import os
    import os.path

    directory_path = './data'  # Replace with your directory path

    # checking whether the training data is ready
    PATH = './training.data'
    if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
        print('Almost Done...')
    else:
        print('training data is being created...')
        open('training.data', 'w')
        color_histogram_feature_extraction.training()

    for filename in os.listdir(directory_path):
        # Check if the file is an image
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # read the test image
            source_image = cv2.imread(os.path.join(directory_path, filename))
            prediction = 'n.a.'
            # get the prediction
            color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
            prediction = knn_classifier.main('training.data', 'test.data')
            if prediction == "fair":
                text = "Bleaching Detected"
            else:
                text = "No Bleaching detected"
            cv2.putText(
                source_image,
                'Prediction: ' + text,
                (15, 45),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                200,
                5
            )
            skintone = (prediction + " color .")
            print(skintone)

            cv2.imshow(f"Output - {filename}", source_image)
            cv2.waitKey()


if __name__ == '__main__':
    main()
