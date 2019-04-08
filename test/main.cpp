#include  <iostream>
#include <string>
#include <map>
using namespace std;

typedef void(*CupFunction)(const string&);
void linemod_train(const string &strConfigFile);
void linemod_recon(const string &strConfigFile);
void linemod_acq(const string &strConfigFile);
//void img_train(const string &strConfigFile);
//void img_recon(const string &strConfigFile);
//void qrc_recon(const string &strConfigFile);
//void obj_recon_visualize(const string &strConfigFile);
//void calibration(const string &cam_id);

static CupFunction GetCupFunction(string& strFuncName);
static void Help();

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        cout << "invalid command line paramter" << endl;
        Help();
        return 0;
    }

    string name = string(argv[1]);
    CupFunction func = GetCupFunction(name);
    if (func == NULL)
    {
        Help();
        return 0;
    }
    func(string(argv[2]));
    return 0;
}

typedef map<string, CupFunction> CupFuncMap;
CupFuncMap gCupFuncMap;

#define RegisterBrewFunction(func) \
namespace { \
class __Registerer_##func { \
 public: /* NOLINT */ \
  __Registerer_##func() { \
    gCupFuncMap[#func] = &func; \
  } \
}; \
__Registerer_##func g_registerer_##func; \
}

RegisterBrewFunction(linemod_train);
RegisterBrewFunction(linemod_recon);
RegisterBrewFunction(linemod_acq);

static CupFunction GetCupFunction(string& strFuncName)
{
    if (gCupFuncMap.count(strFuncName))
    {
        return gCupFuncMap[strFuncName];
    }
    else
    {
        cout << "Unknown action: " << strFuncName << endl;

        return NULL;
    }
}

static void Help()
{
    cout << endl << endl
        << "*********** linemod demo Help ***********" << endl << endl
        << "Usage:" << endl
        << "    linemod [action] [config_file]" << endl << endl;
    cout << "Available actions:" << endl;
    for (CupFuncMap::iterator it = gCupFuncMap.begin(); it != gCupFuncMap.end(); ++it) cout << "    " << it->first;
    cout << endl << endl;
    cout << "********** linemod demo Help ***********" << endl << endl;
}

