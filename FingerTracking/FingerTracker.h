
#ifndef ___FingerTracker_H
#define ___FingerTracker_H

#include "CalibrationData.h"

struct Candidate
{
	Candidate()
		: m_confidence(0.0), m_found(false)
	{
	}

	float		m_confidence;
	cv::Rect	m_windowRect;
	cv::Point   m_fingerPosition;
	bool		m_found;
};

class FingerTracker
{
public:
	FingerTracker();

	void Setup();
	
	void Start();

	Candidate GetCandidate() const;

private:
	void Display(cv::Mat frame, Candidate result) const;
	void Process(cv::Mat frame);
	void LoadInitialParamters();
	cv::Point GetFingerTopPosition(cv::Mat thresholdedFingerFrame) const;
	
private:
	CalibrationData::Ptr	m_calibrationData;
	Candidate				m_currentCandidate;
	std::deque<cv::Point>	m_points;

	int		m_roiSpanX, m_roiSpanY;
	int		m_erosionSize, m_dilationSize;
	float   m_candidateDetecionConfidenceThreshold;
	int		m_backProjectionThreshold;
	
};

#endif // ___FingerTracker_H