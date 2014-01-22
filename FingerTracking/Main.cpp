
#include "stdafx.h"
#include "FingerTracker.h"

int main()
{
	try
	{
		FingerTracker fingerTracker;
		fingerTracker.Setup();
		fingerTracker.Start();
	}
	catch (std::runtime_error const& error)
	{
		std::cout << "Run time error: " << error.what() << std::endl;
	}
	return 0;
}