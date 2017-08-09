/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * MainX.cpp
 *
 *  Created on: Nov 17, 2011
 *      Author: Georg Nebehay
 */

#include "Main.h"

#include "Config.h"
#include "ImAcq.h"
#include "Gui.h"
#include "TLDUtil.h"
#include "Trajectory.h"

#include <vector>
#include <string>
#include <dirent.h>
#include <stdio.h>


using namespace tld;
using namespace cv;

DIR *d;
struct dirent *dir;

//void
//readImgsDir(string path, vector<string> *list_path_imgs){
//	d = opendir(path.c_str());
//	if(d)
//	{
//	while ((dir = readdir(d)) != NULL){
//		string img_path = dir->d_name;
////		std::cout << img_path.substr(img_path.find("."),img_path.length())<< std::endl;
////		if(img_path.substr(img_path.find("."),img_path.length()) == "jpg")
//			list_path_imgs->push_back(img_path);
//
//	}
//
//	}else{
//		std::cout << "erro";
//		exit(0);
//	}
//}



double
getJaccardCoefficient(Rect pred, Rect gt)
{
	double leftCol = pred.x;
	double topRow = pred.y;
	double rightCol = pred.x + pred.width;
	double bottomRow = pred.y + pred.width;

	double gtLeftCol = gt.x;
	double gtTopRow = gt.y;
	double gtRightCol = gt.x + gt.width;
	double gtBottomRow = gt.y + gt.width;

	double jaccCoeff = 0.;

	if (!(leftCol > gtRightCol ||
			rightCol < gtLeftCol ||
			topRow > gtBottomRow ||
			bottomRow < gtTopRow)
	)
	{
		double interLeftCol = std::max<double>(leftCol, gtLeftCol);
		double interTopRow = std::max<double>(topRow, gtTopRow);
		double interRightCol = std::min<double>(rightCol, gtRightCol);
		double interBottomRow = std::min<double>(bottomRow, gtBottomRow);

		const double areaIntersection = (abs(interRightCol - interLeftCol) + 1) * (abs(interBottomRow - interTopRow) + 1);
		const double lhRoiSize = (abs(rightCol - leftCol) + 1) * (abs(bottomRow - topRow) + 1);
		const double rhRoiSize = (abs(gtRightCol - gtLeftCol) + 1) * (abs(gtBottomRow - gtTopRow) + 1);

		jaccCoeff = areaIntersection / (lhRoiSize + rhRoiSize - areaIntersection);
	}
	return jaccCoeff;
}


void readGroundtruth(string video_path, vector<Rect> *ground_truth)
{
	string groundtruth_path = video_path + "/groundtruth.txt";
	FILE* groundtruth_file_ptr = fopen(groundtruth_path.c_str(), "r");
	std::cout <<  groundtruth_path << std::endl;

	while (true)
	{
		// Read the annotation data.
		Rect rect;
		double x, y, width, height;
		const int status = fscanf(groundtruth_file_ptr, "%lf,%lf,%lf,%lf\n",
				&x, &y, &width, &height);

		if (status == EOF)
			break;

		// Increment the frame number.
		//frame_num++;
		rect.x = x;
		rect.y = y;
		rect.width = width;
		rect.height = height;

		ground_truth->push_back(rect);

	} // Process annotation file

	fclose(groundtruth_file_ptr);
}

void Main::doWork()
{
	Trajectory trajectory;
	std::cout << imAcq->imgPath;
    IplImage *img = imAcqGetImg(imAcq);

    Mat grey(img->height, img->width, CV_8UC1);
    cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);

    tld->detectorCascade->imgWidth = grey.cols;
    tld->detectorCascade->imgHeight = grey.rows;
    tld->detectorCascade->imgWidthStep = grey.step;

    std::string path_gt = std::string(imAcq->imgPath);
    vector<Rect> ground_truth;

    readGroundtruth(path_gt, &ground_truth);

	if(showTrajectory)
	{
		trajectory.init(trajectoryLength);
	}

    if(selectManually)
    {

        CvRect box;

        if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
        {
            return;
        }

        if(initialBB == NULL)
        {
            initialBB = new int[4];
        }

        initialBB[0] = box.x;
        initialBB[1] = box.y;
        initialBB[2] = box.width;
        initialBB[3] = box.height;
    }

    FILE *resultsFile = NULL;

    if(printResults != NULL)
    {
        resultsFile = fopen(printResults, "w");
        if(!resultsFile)
        {
            fprintf(stderr, "Error: Unable to create results-file \"%s\"\n", printResults);
            exit(-1);
        }
    }

    bool reuseFrameOnce = false;
    bool skipProcessingOnce = false;

    if(loadModel && modelPath != NULL)
    {
        tld->readFromFile(modelPath);
        reuseFrameOnce = true;
    }
    else if(initialBB != NULL)
    {
        Rect bb = tldArrayToRect(initialBB);

        printf("Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);

        tld->selectObject(grey, &bb);
        skipProcessingOnce = true;
        reuseFrameOnce = true;
    }

    while(imAcqHasMoreFrames(imAcq))
    {
    	int loss = 0;
        double tic = cvGetTickCount();

        if(!reuseFrameOnce)
        {
            cvReleaseImage(&img);
            img = imAcqGetImg(imAcq);

            if(img == NULL)
            {
                printf("current image is NULL, assuming end of input.\n");
                break;
            }

            cvtColor(cvarrToMat(img), grey, CV_BGR2GRAY);
        }

        if(!skipProcessingOnce)
        {
            tld->processImage(cvarrToMat(img));
        }
        else
        {
            skipProcessingOnce = false;
        }

        double jaccard = 0.0001;

        if(tld->currBB!=NULL)
        {
        	Rect tld_bb = *tld->currBB;
        	jaccard = getJaccardCoefficient(tld_bb, ground_truth.at(imAcq->currentFrame));
        }

        if ((jaccard < 0.2) && (jaccard != 0.0001)){
        	loss = 1;
        	*tld->currBB = ground_truth.at(imAcq->currentFrame);
//        	printf("Perdeu!!! Jaccard: %.2f\n", jaccard);
        }

        if(printResults != NULL)
        {
            if(tld->currBB != NULL)
            {
                //fprintf(resultsFile, "%d %.2d %.2d %.2d %.2d %f\n", imAcq->currentFrame - 1, tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height, tld->currConf);
                fprintf(resultsFile, "%.2d,%.2d,%.2d,%.2d,%.2lf,%d,%.2lf\n", tld->currBB->x, tld->currBB->y, tld->currBB->width, tld->currBB->height, jaccard, loss, tld->currConf);
            }
            else
            {
                fprintf(resultsFile, "%d NaN NaN NaN NaN NaN\n", imAcq->currentFrame - 1);
            }
        }
        loss = 0;
        double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();

        toc = toc / 1000000;

        float fps = 1 / toc;

        int confident = (tld->currConf >= threshold) ? 1 : 0;

        if(showOutput || saveDir != NULL)
        {
            char string[128];

            char learningString[10] = "";

            if(tld->learning)
            {
                strcpy(learningString, "Learning");
            }

            sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, tld->currConf, fps, tld->detectorCascade->numWindows, learningString);
            CvScalar yellow = CV_RGB(255, 255, 0);
            CvScalar blue = CV_RGB(0, 0, 255);
            CvScalar black = CV_RGB(0, 0, 0);
            CvScalar white = CV_RGB(255, 255, 255);

            if(tld->currBB != NULL)
            {
                CvScalar rectangleColor = (confident) ? blue : yellow;
                cvRectangle(img, tld->currBB->tl(), tld->currBB->br(), rectangleColor, 8, 8, 0);

				if(showTrajectory)
				{
					CvPoint center = cvPoint(tld->currBB->x+tld->currBB->width/2, tld->currBB->y+tld->currBB->height/2);
					cvLine(img, cvPoint(center.x-2, center.y-2), cvPoint(center.x+2, center.y+2), rectangleColor, 2);
					cvLine(img, cvPoint(center.x-2, center.y+2), cvPoint(center.x+2, center.y-2), rectangleColor, 2);
					trajectory.addPoint(center, rectangleColor);
				}
            }
			else if(showTrajectory)
			{
				trajectory.addPoint(cvPoint(-1, -1), cvScalar(-1, -1, -1));
			}

			if(showTrajectory)
			{
				trajectory.drawTrajectory(img);
			}

            CvFont font;
            cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, .5, .5, 0, 1, 8);
            cvRectangle(img, cvPoint(0, 0), cvPoint(img->width, 50), black, CV_FILLED, 8, 0);
            cvPutText(img, string, cvPoint(25, 25), &font, white);

            if(showForeground)
            {

                for(size_t i = 0; i < tld->detectorCascade->detectionResult->fgList->size(); i++)
                {
                    Rect r = tld->detectorCascade->detectionResult->fgList->at(i);
                    cvRectangle(img, r.tl(), r.br(), white, 1);
                }

            }


            if(showOutput)
            {
                gui->showImage(img);
                char key = gui->getKey();

                if(key == 'q') break;

                if(key == 'b')
                {

                    ForegroundDetector *fg = tld->detectorCascade->foregroundDetector;

                    if(fg->bgImg.empty())
                    {
                        fg->bgImg = grey.clone();
                    }
                    else
                    {
                        fg->bgImg.release();
                    }
                }

                if(key == 'c')
                {
                    //clear everything
                    tld->release();
                }

                if(key == 'l')
                {
                    tld->learningEnabled = !tld->learningEnabled;
                    printf("LearningEnabled: %d\n", tld->learningEnabled);
                }

                if(key == 'a')
                {
                    tld->alternating = !tld->alternating;
                    printf("alternating: %d\n", tld->alternating);
                }

                if(key == 'e')
                {
                    tld->writeToFile(modelExportFile);
                }

                if(key == 'i')
                {
                    tld->readFromFile(modelPath);
                }

                if(key == 'r')
                {
                    CvRect box;

                    if(getBBFromUser(img, box, gui) == PROGRAM_EXIT)
                    {
                        break;
                    }

                    Rect r = Rect(box);

                    tld->selectObject(grey, &r);
                }
            }

            if(saveDir != NULL)
            {
                char fileName[256];
                sprintf(fileName, "%s/%.5d.png", saveDir, imAcq->currentFrame - 1);

                cvSaveImage(fileName, img);
            }
        }

        if(reuseFrameOnce)
        {
            reuseFrameOnce = false;
        }
    }

    cvReleaseImage(&img);
    img = NULL;

    if(exportModelAfterRun)
    {
        tld->writeToFile(modelExportFile);
    }

    if(resultsFile)
    {
        fclose(resultsFile);
    }
}
