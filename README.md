# coeus engine

CUDA-based rendering and physics engine.

# Import Models

In order to avoid the excessive growth of the repo we decided to load all the models on a Drive. <br />
Here you can find all the models used: [Assets Folder](https://drive.google.com/drive/u/1/folders/12JN1tYfdxabtCbQ0thMZrkp1TML8EXkX) <br />
In order to download the Model folder to your local machine just run the 'downloadModels.ps1' powershell script.

# Load Models

To load a model with textures, make sure that in the same directory as the model, there is a folder called "Textures" which contains all used textures.
If model's textures are somewhere else, a full black model will be loaded.

# CUDA Compilation

In order to perform CUDA compilation run the powershell script provided inside the CUDA folder. <br />
Once the .lib and the .dll files are generated you might add the flag -l{NameOfThe".lib"FileWithoutExtension} to Tasks.json args.

# Compilation

Just simply press Ctrl + Shift + B in order to fully (re)compile and run the project.
If you wish to compile only some components in order to avoid the full recompilation use the script <br />
PartialMakefile.bat with subsequent parameters -> ex.: PartialMakefile.bat camera model <br />
this command will recompile only Camera.cpp and Model.cpp classes resulting in a fast paced compilation.

# Serializable Component Creation
In order to create a Serializable Component you have to use a two step registration. <br />
- First you have to register the class Ancestors (You have to always specify ```System::Component``` as an ancestor)
- After this you can go ahead and do the registration in the *.cpp* file. Here a sample of the registration for the Mesh Component:
```
    SERIALIZE_CLASS
    {
        System::SerializableClass::registerClass<Mesh>();
    }
```

After the registration you may want to implement the methods of the ```System::Component``` interface that effectively serialize, deserialize and read a component
into our Coeus Engine.

Here an example with a DummyComponent, in the *.cpp* file:
```
void DummyComponent::showComponentFieldsInEditor()
{
    ImGui::InputFloat(NAMEOF(asd), &asd);
    ImGui::InputInt(NAMEOF(var), &var);
}

void DummyComponent::serialize(YAML::Emitter& out)
{
    out << YAML::Key << this->toString();
    out << YAML::BeginMap;
        out << YAML::Key << NAMEOF(asd) << YAML::Value << this->asd;
        out << YAML::Key << NAMEOF(var) << YAML::Value << this->var;
    out << YAML::EndMap; 
}

System::Component* DummyComponent::deserialize(YAML::Node& node)
{
    auto component = node[this->toString()];
    this->asd = component["asd"].as<int>();
    this->var = component["var"].as<int>();

    return this;
}
```

> In order to Serialize and Deserialize you should be aware of the syntax of the [YAML-cpp](https://github.com/jbeder/yaml-cpp/) library.
