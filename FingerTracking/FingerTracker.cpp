
#include "stdafx.h"

#include "FingerTracker.h"
#include <chrono>

using cv::Mat;
using cv::MatND;
using cv::Rect;
using cv::Point;
using cv::Size;
using cv::MORPH_RECT;
using cv::Scalar;
using cv::VideoCapture;
using cv::rectangle;
using cv::imshow;
using cv::namedWindow;
using cv::createTrackbar;
using cv::createButton;
using cv::NORM_MINMAX;
using cv::minMaxLoc;
using cv::putText;
using cv::getTextSize;
using cv::setMouseCallback;
using cv::EVENT_LBUTTONUP;
using cv::magnitude;

#define ENABLE_DEBUG_WINDOWS 0
#define ENABLE_LINE_DRAWING 1
const int LINE_HISTORY = 200;

FingerTracker::FingerTracker(void)
{
	LoadInitialParamters();
}

void FingerTracker::LoadInitialParamters()
{
	m_roiSpanX = 50;
	m_roiSpanY = 50;
	m_erosionSize = 1;
	m_dilationSize = 2;
	m_candidateDetecionConfidenceThreshold = 0.98f;
	m_backProjectionThreshold = 10;

	m_calibrationBottomLowerThd = 60; //< Experiment 50-70
	m_calibrationBottomUpperThd = 90;
	m_calibrationMiddleLowerThd = 60;
	m_calibrationMiddleUppperThd = 90;
	m_calibrationTopLowerThd = 30;
	m_calibrationTopUpperThd = 60;

	m_calibrationDiffThreshold = 15;
}

void TrackBarCallback(int state, void* userdata)
{
	auto value = reinterpret_cast<int*>(userdata);
	if (value)
	{
		*value = 1;
	}
}

void MouseCallback(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONUP)
	{
		auto value = reinterpret_cast<int*>(userdata);
		if (value)
		{
			*value = 1;
		}
	}
}

void FingerTracker::Setup()
{
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		throw std::runtime_error("Could not start camera capture");
	}

	int windowSize = 25;
	int Xpos = 200;
	int Ypos = 50;
	int update = 0;
	int buttonClicked = 0;
	namedWindow("RGB", CV_WINDOW_AUTOSIZE);
	createTrackbar("X", "RGB",  &Xpos, 320, TrackBarCallback, (void*)&update);
	createTrackbar("Y", "RGB",  &Ypos, 240, TrackBarCallback, (void*)&update);
	createTrackbar("Size", "RGB",  &windowSize, 100, TrackBarCallback, (void*)&update);
	setMouseCallback("RGB", MouseCallback, (void*)&buttonClicked);
	Mat fingerWindowBackground, fingerWindowBackgroundGray;

	m_calibrationData.reset(new CalibrationData());
	bool ticking = false;
	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    while (true)
	{  		
		Mat frame, frameHSV;

		if (capture.read(frame))
		{
			flip(frame, frame, 1);

			pyrDown(frame, frame, Size(frame.cols / 2, frame.rows / 2));

			Rect fingerWindow(Point(Xpos, Ypos), Size(windowSize, windowSize*3));
			if (Xpos + windowSize >= frame.cols || Ypos + windowSize*3 >= frame.rows)
			{
				windowSize = 20;
				Xpos = 200;
				Ypos = 50;
				update = 0;
			}
			else if (buttonClicked == 1)
			{
				frame(fingerWindow).copyTo(fingerWindowBackground);
				cvtColor(fingerWindowBackground, fingerWindowBackgroundGray, CV_BGR2GRAY);
				buttonClicked  = 0;
				update = 0;
				cvDestroyAllWindows();
			}

			if (fingerWindowBackgroundGray.rows && !m_calibrationData->m_ready)
			{
				Mat diff, thd;
				absdiff(frame(fingerWindow), fingerWindowBackground, diff);
				std::vector<Mat> ch;
				split(diff, ch);
				threshold(ch[0], ch[0], m_calibrationDiffThreshold, 255, 0);
				threshold(ch[1], ch[1], m_calibrationDiffThreshold, 255, 0);
				threshold(ch[2], ch[2], m_calibrationDiffThreshold, 255, 0);
				thd = ch[0];
				add(thd, ch[1], thd);
				add(thd, ch[2], thd);
				medianBlur(thd, thd, 5);

				Mat top, middle, bottom;
				Rect r1 = Rect(0, 0, thd.cols, thd.rows/3);
				Rect r2 = Rect(0, thd.rows / 3 + 1, thd.cols, thd.rows/3);
				Rect r3 = Rect(0, thd.rows * 2 / 3 + 1, thd.cols, thd.rows -  thd.rows * 2 / 3 - 1);
				top = thd(r1);
				middle = thd(r2);
				bottom = thd(r3);			

				auto percentageTop = countNonZero(top) * 100.0 / top.size().area();
				auto percentageMiddle = countNonZero(middle) * 100.0 / middle.size().area();
				auto percentageBottom = countNonZero(bottom) * 100.0 / bottom.size().area();

				bool topReady = false;
				bool middleReady = false;
				bool bottomReady = false;

				Scalar c1, c2, c3;
				if (percentageTop > m_calibrationTopLowerThd && percentageTop < m_calibrationTopUpperThd)
				{
					topReady = true;
					c1 = Scalar(0, 255, 255);
				}
				else
				{
					c1 = Scalar(0, 0, 255);
				}

				if (percentageMiddle > m_calibrationMiddleLowerThd && percentageMiddle < m_calibrationMiddleUppperThd)
				{
					middleReady = true;					
					c2 = Scalar(0, 255, 255);
				}
				else
				{
					c2 = Scalar(0, 0, 255);
				}

				if (percentageBottom > m_calibrationBottomLowerThd && percentageBottom < m_calibrationBottomUpperThd)
				{
					bottomReady = true;
					c3 = Scalar(0, 255, 255);
				}
				else
				{
					c3 = Scalar(0, 0, 255);
				}

				bool readyToGo = false;
				if (middleReady && topReady && bottomReady)
				{
					c1 = Scalar(0, 255, 0);
					c2 = Scalar(0, 255, 0);
					c3 = Scalar(0, 255, 0);
					if (!ticking)
					{
						start = std::chrono::system_clock::now();
						ticking = true;
					}
					if (std::chrono::system_clock::now() - start > std::chrono::seconds(1))
					{
						readyToGo = true;
					}
				}
				else
				{
					ticking = false;
				}

#if ENABLE_DEBUG_WINDOWS
				std::stringstream ss;
				ss << percentageTop << ", " << percentageMiddle << ", " << percentageBottom;
				putText(frame, ss.str(), Point(0, getTextSize(ss.str(), 0, 0.5, 1, nullptr).height), 0, 0.5, Scalar::all(255), 1);
				cv::imshow("Thresholded", thd);
#endif

				if (percentageTop >= m_calibrationTopUpperThd && percentageBottom >= m_calibrationBottomUpperThd && percentageMiddle >= m_calibrationMiddleUppperThd)
				{
					putText(frame, "Move finger away from camera", Point(0, getTextSize("Move finger away from camera", 0, 0.5, 1, nullptr).height), 0, 0.5, Scalar::all(255), 1);
				}
				else if (percentageTop <= m_calibrationTopLowerThd && percentageBottom <= m_calibrationBottomLowerThd && percentageMiddle <= m_calibrationMiddleLowerThd)
				{
					putText(frame, "Move finger closer to camera", Point(0, getTextSize("Move finger closer to camera", 0, 0.5, 1, nullptr).height), 0, 0.5, Scalar::all(255), 1);
				}
				
				if (readyToGo)
				{
					Mat framePatchHSV;
					cvtColor(frame(fingerWindow), framePatchHSV, CV_BGR2HSV);
					cvtColor(frame, frameHSV, CV_BGR2HSV);

					MatND hist;
					calcHist(&framePatchHSV, 1, m_calibrationData->m_channels, thd, hist, 2, m_calibrationData->m_histSize, (const float**)m_calibrationData->m_ranges, true, false);
					m_calibrationData->m_hist = hist;
					normalize(m_calibrationData->m_hist, m_calibrationData->m_hist, 0, 255, NORM_MINMAX, -1, Mat());

#if ENABLE_DEBUG_WINDOWS

					double maxVal=0;
					minMaxLoc(m_calibrationData->m_hist, 0, &maxVal, 0, 0);
					int scale = 10;
					Mat histImg = Mat::zeros(m_calibrationData->m_sbins*scale, m_calibrationData->m_hbins*10, CV_8UC3);
					for( int h = 0; h < m_calibrationData->m_hbins; h++)
					{
						for( int s = 0; s < m_calibrationData->m_sbins; s++ )
						{
							float binVal = m_calibrationData->m_hist.at<float>(h, s);
							int intensity = cvRound(binVal*255/maxVal);
							rectangle( histImg, Point(h*scale, s*scale),
										Point( (h+1)*scale - 1, (s+1)*scale - 1),
										Scalar::all(intensity),
										CV_FILLED );
						}
					}

					imshow("H-S Histogram", histImg);
#endif
					m_calibrationData->m_ready = true;
					frame(fingerWindow).copyTo(m_calibrationData->m_fingerPatch);
					m_calibrationData->m_fingerRect = fingerWindow;
					m_currentCandidate.m_windowRect = fingerWindow;
					m_currentCandidate.m_fingerPosition = fingerWindow.tl();
					return;
				}

				rectangle(frame, r1.tl() + fingerWindow.tl(), r1.br() + fingerWindow.tl(), c1);
				rectangle(frame, r2.tl() + fingerWindow.tl(), r2.br() + fingerWindow.tl(), c2);
				rectangle(frame, r3.tl() + fingerWindow.tl(), r3.br() + fingerWindow.tl(), c3);
				imshow("Calibration", frame);
			}
			else
			{
				int baseline = 0;
				putText(frame, "Adjust calibration window, click when ready", Point(0, getTextSize("Adjust calibration window", 0, 0.4, 2, &baseline).height), 0, 0.4, Scalar::all(255), 1);
				rectangle(frame, fingerWindow.tl(), fingerWindow.br(), Scalar(0, 0, 255));
				imshow("RGB", frame);
			}
			auto key = cvWaitKey(10);
			if (char(key) == 27)
			{
				break;
			}
		}
	}
	capture.release();
}

void FingerTracker::Display(Mat frame, Candidate candidate) const
{
	if (candidate.m_found)
	{
		rectangle(frame, candidate.m_windowRect.tl(), candidate.m_windowRect.br(), Scalar(0, 255, 255));
		rectangle(frame, candidate.m_fingerPosition - Point(2,2), candidate.m_fingerPosition + Point(2,2), Scalar(255, 0, 0), 2);
	}
#if ENABLE_LINE_DRAWING
	Point prev(-1,-1);
	for (auto point : m_points)
	{
		if (prev.x != -1)
		{
			line(frame, prev, point, Scalar(0, 255, 0));
		}
		prev = point;
	}

#endif
	imshow("Camera", frame);
}

void FingerTracker::Start()
{
	cvDestroyAllWindows();
	VideoCapture capture(0);
	if (!capture.isOpened())
	{
		throw std::runtime_error("Could not start camera capture");
	}

	Mat frame;
    while (true)
	{
		if (capture.read(frame))
		{
			if (frame.rows && frame.cols)
			{
				flip(frame, frame, 1);
				pyrDown(frame, frame, Size(frame.cols / 2, frame.rows / 2));

				Process(frame);

				auto const& candidate = GetCandidate();

#if ENABLE_LINE_DRAWING
				if (candidate.m_found)
				{
					m_points.push_back(candidate.m_fingerPosition);
				}
				if (m_points.size() > LINE_HISTORY)
				{
					m_points.pop_front();
				}
#endif

				Display(frame, candidate);
			}
		}
					
		auto key = cvWaitKey(10);
		if (char(key) == 27)
		{
			break;
		}
	}
	capture.release();
}

void FingerTracker::Process(Mat frame)
{
#if ENABLE_DEBUG_WINDOWS
	Mat img_display;
	frame.copyTo(img_display);
#endif
	
	//	Process only Region of Interest i.e. region around current finger position
	Rect roi = Rect(		
		Point(std::max(m_currentCandidate.m_windowRect.tl().x - m_roiSpanX, 0), std::max(m_currentCandidate.m_windowRect.tl().y - m_roiSpanY, 0)), 		
		Point(std::min(m_currentCandidate.m_windowRect.tl().x + m_roiSpanX + m_calibrationData->m_fingerPatch.cols, frame.cols), 
			  std::min(m_currentCandidate.m_windowRect.tl().y + m_roiSpanY + m_calibrationData->m_fingerPatch.rows, frame.rows)));

	Mat frameRoi;
	frame(roi).copyTo(frameRoi);

	//================TEMPLATE MATCHING
	int result_cols =  frameRoi.cols - m_calibrationData->m_fingerPatch.cols + 1;
	int result_rows = frameRoi.rows - m_calibrationData->m_fingerPatch.rows + 1;
	assert(result_cols > 0 && result_rows > 0);

	Mat scoreMap;
	scoreMap.create(result_cols, result_rows, CV_32FC1);

	//	Compare current frame roi region to known candidate
	//	Using OpenCV matchTemplate function with correlation coefficient matching method
	matchTemplate(frameRoi, m_calibrationData->m_fingerPatch, scoreMap, 3);

	//================HISTOGRAM BACK PROJECTION
	MatND backProjection;
	Mat frameHSV;
	cvtColor(frameRoi, frameHSV, CV_BGR2HSV);

	calcBackProject(&frameHSV, 1, m_calibrationData->m_channels, m_calibrationData->m_hist, backProjection, (const float**)(m_calibrationData->m_ranges), 1, true);

	Mat backProjectionThresholded;
	threshold(backProjection, backProjectionThresholded, m_backProjectionThreshold, 255, 0);
	erode(backProjectionThresholded, backProjectionThresholded, getStructuringElement(MORPH_RECT, Size(2 * m_erosionSize + 1, 2 * m_erosionSize + 1), Point(m_erosionSize, m_erosionSize)));
	dilate(backProjectionThresholded, backProjectionThresholded, getStructuringElement(MORPH_RECT, Size(2 * m_dilationSize + 1, 2 * m_dilationSize + 1), Point(m_dilationSize, m_dilationSize)));

	Mat backProjectionThresholdedShifted;
	Rect shifted(Rect(m_calibrationData->m_fingerPatch.cols - 1, m_calibrationData->m_fingerPatch.rows - 1, scoreMap.cols, scoreMap.rows));
	backProjectionThresholded(shifted).copyTo(backProjectionThresholdedShifted);

	Mat maskedOutScoreMap(scoreMap.size(), CV_8U);
	scoreMap.copyTo(maskedOutScoreMap, backProjectionThresholdedShifted);

	//====================Localizing the best match with minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;
	minMaxLoc(maskedOutScoreMap, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = maxLoc + roi.tl();

	m_currentCandidate.m_confidence = static_cast<float>(maxVal);

	if (maxVal > m_candidateDetecionConfidenceThreshold)
	{
		m_currentCandidate.m_found = true;
		m_currentCandidate.m_windowRect = Rect(matchLoc, Point(matchLoc.x + m_calibrationData->m_fingerPatch.cols , matchLoc.y + m_calibrationData->m_fingerPatch.rows));		

		//================Find finger position
		Mat fingerWindowThresholded;
		backProjectionThresholded(Rect(maxLoc, Point(maxLoc.x + m_calibrationData->m_fingerPatch.cols , maxLoc.y + m_calibrationData->m_fingerPatch.rows))).copyTo(fingerWindowThresholded);
		
		m_currentCandidate.m_fingerPosition = GetFingerTopPosition(fingerWindowThresholded) + matchLoc;

#if ENABLE_DEBUG_WINDOWS
		rectangle(img_display, m_currentCandidate.m_windowRect.tl(), m_currentCandidate.m_windowRect.br(), Scalar(255,0,0), 2, 8, 0 );
		rectangle(scoreMap, m_currentCandidate.m_windowRect.tl(), m_currentCandidate.m_windowRect.br(), Scalar::all(0), 2, 8, 0 );
		rectangle(img_display, m_currentCandidate.m_fingerPosition, m_currentCandidate.m_fingerPosition + Point(5,5), Scalar(255,0,0));
#endif
	}
	else
	{
		m_currentCandidate.m_found = false;
	}

#if ENABLE_DEBUG_WINDOWS
	std::stringstream ss;
	ss << maxVal;
	putText(img_display, ss.str(), Point(50, 50), 0, 0.5, Scalar(255,255,255), 1);

	imshow("Overlays", img_display);
	imshow("Results", scoreMap);
	imshow("ResultMasked", maskedOutScoreMap);
#endif
}

Point FingerTracker::GetFingerTopPosition(Mat thresholdedFingerFrame) const
{
	uchar* ptr = thresholdedFingerFrame.data;
	for (int i = 0; i < thresholdedFingerFrame.rows * thresholdedFingerFrame.cols; i++)
	{
		if (*ptr++ == 255)
		{
			int y = i / thresholdedFingerFrame.cols;
			int x = i % thresholdedFingerFrame.cols;
			return Point(x,y);
		}
	}
	return Point(0,0);
}

Candidate FingerTracker::GetCandidate() const
{
	return m_currentCandidate;
}


