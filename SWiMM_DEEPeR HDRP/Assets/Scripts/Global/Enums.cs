
using System.Collections.Generic;

public static class Enums
{
    public enum E_SceneIndices
    {
        PERSISTENT_SCENE = 0,
        MAIN_MENU = 1,
        SIMULATION = 2,
        EXIT = 3
    }

    public enum E_LevelType
    {
        MANUAL = 0,
        TRAINING = 1,
        INFERENCE = 2
    }

    public enum E_InitialisationStage
    {
        INITIALISING_NPCS = 0,
        SPAWNING_NPCS = 1
    }

    public enum E_Protocol
    {
        UDP = 0,
        TCP = 1
    }

    public static Dictionary<string, E_Protocol> protocol_mapping = new Dictionary<string, E_Protocol>
    {
        { "udp", Enums.E_Protocol.UDP },
        { "tcp", Enums.E_Protocol.TCP }
    };

    public enum E_Action_Inference
    {
        ON_RECEIVE = 0,
        MAINTAIN = 1,
        FREEZE = 2,
        MAINTAIN_FREEZE = 3
    }

    public static Dictionary<string, E_Action_Inference> action_inference_mapping = new Dictionary<string, E_Action_Inference>
    {
        { "onReceive", E_Action_Inference.ON_RECEIVE },
        { "maintain", E_Action_Inference.MAINTAIN },
        { "freeze", E_Action_Inference.FREEZE },
        { "maintainFreeze", E_Action_Inference.MAINTAIN_FREEZE },
    };

    public enum E_Automation_Actions
    {
        TO_OPTIONS = 0,
        SERVER_CONNECT = 1,
        TO_MENU = 2,
        START_TRAINING = 3
    }

    public enum E_Rover_Dive_Mode
    {
        MANUAL = 0,
        DEPTH_HOLD = 1,
        STABILIZE = 2
    }
}
