using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
using Task = System.Threading.Tasks.Task;

public class Server
{
    public int resets_received = 0;
    public int episode_num = -1;
    public int obsv_num = 0;
    public bool first_observation_sent = false;
    byte[] receive_buffer;

    public TcpClient tcp_client = null;
    public NetworkStream stream = null;

    public UdpClient udp_client = null;


    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ServerConfig
    {
        public RoverConfig roverConfig;
        public EnvironmentConfig envConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct JsonMessage
    {
        public string msgType;
        public PayloadDataToReceive payload;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct PayloadDataToReceive
    {
        public int episode_num;
        public int action_num;
        public ServerConfig serverConfig;
        public JsonControls jsonControls;
        public GameObjectPosition[] objectPositions;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct GameObjectPosition
    {
        public string object_name;
        public float[] position;
        public float[] rotation;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ActionSpaceConfig
    {
        public int dimensions;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct RoverConfig
    {
        public MotorConfig motorConfig;
        public CameraConfig camConfig;
        public StructureConfig structureConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct MotorConfig
    {
        public float stabilityThreshold;
        public float stabilityForce;
        public float[] linearThrustPower;
        public float[] angularThrustPower;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct CameraConfig
    {
        public int[] resolution;
        public int fov;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct StructureConfig
    {
        [Range(0f, 1.2f)]
        public float ballastMass;   // rover spec, 6x200g masses
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct EnvironmentConfig
    {
        public string actionInference;
        public float optD;
        public float maxD;
        public FaunaConfig faunaConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct FaunaConfig
    {
        public float spawnTimer;
        public float spawnContainerRatio;
        public BoidGroup[] boidGroups;
        public AIGroup[] aiGroups;
        public float spawnRadius;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Buffers
    {
        public int channel_action;
        public int channel_observation;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct DataToSend
    {
        public string msg_type;
        public PayloadDataToSend payload;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct PayloadDataToSend
    {
        public int obsv_num;
        public TelemetryData telemetry_data;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct TelemetryData
    {
        public byte[] jpg_image;
        public float[] position;
        public string[] collision_objects;
        public float[] fwd;
        public TargetObject[] targets;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct TargetObject
    {
        public float[] position;
        public float[] fwd;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct JsonControls
    {
        public float swayThrust;
        public float heaveThrust;
        public float surgeThrust;
        public float pitchThrust;
        public float yawThrust;
        public float rollThrust;
        public float manualMode;
        public float stabilizeMode;
        public float depthHoldMode;
    }

    /* Configs/Messages from server */
    public JsonMessage json_server_config;

    public string json_str_obsv;

    IPAddress ipAddress;
    IPEndPoint ipEndPoint;

    public Server()
    {
        json_str_obsv = null;
    }

    public bool IsConnectionValid()
    {
        return udp_client != null ? true : stream != null;
    }

    public void Disconnect()
    {
        if(udp_client != null)
        {
            udp_client.Close();
        }
        else
        {
            tcp_client.Close();
        }

        stream = null;
        return;
    }

    public Tuple<Server, Exception> Connect(SimulationManager.NetworkConfig network_config, string ip, int port)
    {
        try
        {
            ipAddress = IPAddress.Parse(ip);
            ipEndPoint = new IPEndPoint(ipAddress, port);

            if (Enums.protocol_mapping[network_config.protocol] == Enums.E_Protocol.UDP)
            {
                udp_client = new UdpClient();
                udp_client.Client.ReceiveBufferSize = network_config.buffers.channel_action;
                udp_client.Client.SendBufferSize = network_config.buffers.channel_observation;
                udp_client.Connect(ipEndPoint);
                return new Tuple<Server, Exception>(this, null);
            }

            else
            {
                tcp_client = new TcpClient
                {
                    ReceiveBufferSize = network_config.buffers.channel_action,
                    SendBufferSize = network_config.buffers.channel_observation,
                };

                receive_buffer = new byte[tcp_client.ReceiveBufferSize];
                tcp_client.Connect(ipEndPoint);
                stream = tcp_client.GetStream();
                return new Tuple<Server, Exception>(this, null);
            }
        }

        catch (Exception ex)
        {
            Debug.LogException(ex);
            return new Tuple<Server, Exception>(null, ex);
        }

    }

    public static async Task WaitUntilAsync(Func<bool> predicate, int sleep = 1 / 120)
    {
        while (!predicate())
        {
            await Task.Delay(sleep);
        }
    }

    private bool DataReadyToSend()
    {
        return json_str_obsv != null;
    }

    //public void CastAndOverride<T, P>(JsonMessage<T> data, ref JsonMessage<P> param)
    //{
    //    param = (JsonMessage<P>)(object)(T)(object)data;
    //    param.is_overriden = true;
    //}

    public async Task<GameEvent> ContinueReadWrite() 
    {
        try
        {
            json_str_obsv = JsonConvert.SerializeObject(new DataToSend
            {
                msg_type = "connection_request",
            });

            while (IsConnectionValid())
            {
                /* Writing */
                await WaitUntilAsync(DataReadyToSend);

                DataToSend current_msg = JsonConvert.DeserializeObject<DataToSend>(json_str_obsv);

                current_msg.payload.obsv_num = obsv_num;

                UnityMainThreadDispatcher.Instance().Enqueue(FireSentEvents(current_msg));

                string update_json_str = JsonConvert.SerializeObject(current_msg);

                try
                {
                    if (SimulationManager._instance.debug_logs)
                    {
                       Task t = File.WriteAllTextAsync(Path.GetFullPath(Path.Combine(SimulationManager._instance.packets_sent_dir.FullName, "episode_" + episode_num.ToString() + "_observation_" + current_msg.payload.obsv_num.ToString() + ".json")), update_json_str);
                    }

                    //Debug.Log("Sending: " + update_json_str);
                    byte[] _data = Encoding.UTF8.GetBytes(update_json_str);

                    if(udp_client != null)
                    {
                        await udp_client.SendAsync(_data, _data.Length);        // Have to send connectionless, don't know who's listening
                    }
                    else
                    {
                        await stream.WriteAsync(_data, 0, _data.Length);
                    }

                    obsv_num++;
                    json_str_obsv = null;
                }

                catch (Exception e)
                {
                    Debug.LogException(e);
                }

                /* Reading */
                string current_json_action = null; 

                if (udp_client != null)
                {
                    byte[] receiveBytes = udp_client.Receive(ref ipEndPoint);
                    current_json_action = Encoding.Default.GetString(receiveBytes);
                }
                else
                {
                    int bytes_read = await stream.ReadAsync(receive_buffer, 0, receive_buffer.Length);

                    if (bytes_read == 0)
                    {
                        await Task.Yield();
                        break;
                    }

                    current_json_action = Encoding.Default.GetString(receive_buffer);
                    Array.Clear(receive_buffer, 0, receive_buffer.Length);
                }

                if (current_json_action != null)
                {
                    //Debug.Log("Received: " + current_json_action);
                    JsonMessage message = JsonConvert.DeserializeObject<JsonMessage>(current_json_action);

                    if(message.msgType == "reset_episode")
                    {
                        obsv_num = 0;

                        if (SimulationManager._instance.debug_logs)
                        {
                            Utils.CleanAndCreateDirectories(new Dictionary<DirectoryInfo, bool>()
                            {
                                { SimulationManager._instance.image_dir, true },
                                { SimulationManager._instance.packets_sent_dir, true },
                                { SimulationManager._instance.packets_received_dir, true }
                            });
                        }
                    }

                    episode_num = message.payload.episode_num;

                    if (SimulationManager._instance.debug_logs)
                    {
                        Task t = File.WriteAllTextAsync(Path.GetFullPath(Path.Combine(SimulationManager._instance.packets_received_dir.FullName, "episode_" + episode_num.ToString() + "_packet_" + message.payload.action_num.ToString() + ".json")), current_json_action);
                    }

                    switch (message.msgType)
                    {
                        case "process_server_config":
                            json_server_config = message;
                            break;
                    }

                    UnityMainThreadDispatcher.Instance().Enqueue(FireServerEvents(message));
                }
            }

            /* TCP is no longer good */
            Disconnect();
            return null;
        }
        
        catch(Exception e)
        {
            Debug.LogException(e);
            Disconnect();
            return null;
        }
    }

    private IEnumerator FireSentEvents(DataToSend msg)
    {
        SimulationManager._instance.event_master.sent_event.Raise(msg);
        yield return null;
    }

    private IEnumerator FireServerEvents(JsonMessage msg)
    {
        SimulationManager._instance.event_master.server_event.Raise(msg);
        yield return null;
    }
}
