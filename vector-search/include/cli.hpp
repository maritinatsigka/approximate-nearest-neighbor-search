#pragma once
#include <string>
#include <unordered_map>

//Command-line parser for flags
struct CLI{
    std::unordered_map<std::string, std::string> args;

    CLI(int argc, char** argv){
        for (int i = 1; i < argc; ++i){
            std::string token = argv[i];

            if (token.size() > 1 && token[0] == '-'){  //flag token
                std::string key = token;    
                std::string value = "true";  //default boolean

                if (i + 1 < argc && argv[i + 1][0] != '-'){  //negative numbers like "-3" will not be captured here
                    value = argv[++i];
                }

                args[key] = value;  //overwrite on dup
            }                       //else: ignore positional
        }
    }

    bool has(const std::string& key) const{  //check if a flag exists
        return args.find(key) != args.end();
    }

    std::string get(const std::string& key, const std::string& def = "") const{ //get string value or default
        auto it = args.find(key);
        return (it == args.end()) ? def : it->second;
    }

    int geti(const std::string& key, int def) const{  //get int value or default
        auto it = args.find(key);
        if (it == args.end()){
            return def;
        }
        try { return std::stoi(it->second); } catch(...) { return def; }
    }

    double getd(const std::string& key, double def) const {  //get double value or default
        auto it = args.find(key);
        if (it == args.end()){
            return def;
        }
        try { return std::stod(it->second); } catch(...) { return def; }
    }
};
