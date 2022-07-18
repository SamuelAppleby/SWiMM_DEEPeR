using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Fog : MonoBehaviour
{
    private bool server_config_processed = false;

    private void OnPostRender()
    {
        //if (SimulationManager._instance.server != null && SimulationManager._instance.server.server_config.is_overridden && !server_config_processed)
        //{
        //    RenderSettings.fogMode = FogMode.Exponential;
        //    RenderSettings.fog = SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogOn;
        //    RenderSettings.fogDensity = SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogDensity;
        //    RenderSettings.fogColor = new Color(SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogColour[0], SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogColour[1],
        //        SimulationManager._instance.server.server_config.payload.envConfig.fogConfig.fogColour[2]);

        //    server_config_processed = true;
        //}
    }
}
