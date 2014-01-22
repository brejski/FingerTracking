
#ifndef ___CalibrationData_H
#define ___CalibrationData_H

struct CalibrationData
{
	typedef std::shared_ptr<CalibrationData> Ptr;
	cv::MatND	m_hist;

	int  		m_channels[2];
	int			m_histSize[2];
	int			m_hbins;
	int			m_sbins;
	float		m_hranges[2];
	float		m_sranges[2];
	float*		m_ranges[2];
	bool		m_ready;
	cv::Mat		m_fingerPatch;
	cv::Rect	m_fingerRect;

	CalibrationData()
	{
		m_ready = false;
		m_channels[0] = 0;
		m_channels[1] = 1;			
		m_hbins = 30;
		m_sbins = 32;
		m_histSize[0] = m_hbins;
		m_histSize[1] = m_sbins;
		m_hranges[0] = 0;
		m_hranges[1] = 180;
		m_sranges[0] = 0;
		m_sranges[1] = 256;
		m_ranges[0] = m_hranges;
		m_ranges[1] = m_sranges;
	}
};

#endif // ___CalibrationData_H