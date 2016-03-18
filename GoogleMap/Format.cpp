#include <iostream>
#include <string>
#include <fstream>

using namespace std;

int main(int argc, char** argv) {
	int status = 0;
	if(argc == 1) {
		cerr << "No file specified. Please specify a filename" << endl;
		status = 1;
	}

	for (int i = 1; i < argc; ++i) {
		string line = "";
		string filename = argv[i], new_filename = "new" + filename;
		ifstream infile;
		ofstream myfile;
		
		infile.open(filename);
		if(infile.fail()) {
			status = 1;
			cerr << filename << "： No Such File Exist" << endl;
		}
		else{
			myfile.open (new_filename);
			while(getline(infile, line)) {
				while(line.find('(') != string::npos) {
					auto index1 = line.find('(');
					line[index1] = '[';
				}
				while(line.find('(') != string::npos) {
					auto index1 = line.find(')');
					line[index1] = ']';
				}
				myfile << line << "\n";
			}
			infile.close();
			myfile.close();
		}
	}


	return status;
}