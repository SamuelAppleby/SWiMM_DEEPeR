using Newtonsoft.Json;
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;
using Task = System.Threading.Tasks.Task;

public class Server
{
    public bool server_crash = false;

    public int observations_sent = 0;

    public int actions_received = 0;

    public int resets_received = 0;

    public bool first_observation_sent = false;

    byte[] receive_buffer;

    public NetworkStream stream = null;

    public TcpClient tcp_client = null;

    public UdpClient udp_client = null;


    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ServerConfig
    {
        public LearningConfig learningConfig;
        public RoverConfig roverConfig;
        public EnvironmentConfig envConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct JsonMessage
    {
        public string msgType;
        public Payload payload;
        public bool is_overriden;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Payload
    {
        public ServerConfig serverConfig;
        public JsonControls jsonControls;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct ActionSpaceConfig
    {
        public int dimensions;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct LearningConfig
    {
        public ActionSpaceConfig actionSpaceConfig;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct RoverConfig
    {
        public CameraConfig camConfig;
        public StructureConfig structureConfig;
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
        public float totalBuoyantForce;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct EnvironmentConfig
    {
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
        public int server_send_buffer_size_kb;
        public int client_receive_buffer_size_kb;
        public int server_receive_buffer_size_kb;
        public int client_send_buffer_size_kb;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct DataToSend
    {
        public string msg_type;
        public Telemetary_Data payload;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct Telemetary_Data
    {
        public int? sequence_num;       // nullify for ints, not native to newtonsoft
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
        public float lateralThrust;
        public float forwardThrust;
        public float verticalThrust;
        public float pitchThrust;
        public float yawThrust;
        public float rollThrust;
        public float depthHoldMode;
    }

    [JsonObject(ItemNullValueHandling = NullValueHandling.Ignore)]
    public struct DatalessMessage
    {
    }

    /* Configs/Messages from server */
    public JsonMessage json_server_config;

    public JsonMessage json_rover_controls;

    public JsonMessage json_reset_episode;

    public JsonMessage json_end_simulation;

    public DataToSend? last_obsv;

    public DataToSend? obsv;

    public Server()
    {
        obsv = null;
    }

    public bool IsTcpGood()
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
        server_crash = true;
        return;
    }

    public async Task<Exception> Connect(SimulationManager.NetworkConfig network_config, string ip, int port)
    {
        if(network_config.e_protocol == Protocol.UDP)
        {
            udp_client = new UdpClient();

            try
            {
                udp_client.Connect(ip, port);
                return null;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                return ex;
            }
        }
        else
        {
            tcp_client = new TcpClient
            {
                ReceiveBufferSize = network_config.buffers.client_receive_buffer_size_kb * 1024,
                SendBufferSize = network_config.buffers.client_send_buffer_size_kb * 1024
            };

            receive_buffer = new byte[tcp_client.ReceiveBufferSize];

            try
            {
                await tcp_client.ConnectAsync(ip, port);
                stream = tcp_client.GetStream();
                return null;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                return ex;
            }
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
        return obsv != null && last_obsv == null;
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
            obsv = new DataToSend
            {
                msg_type = "connection_request",
                payload = new Telemetary_Data { }
            };

            while (IsTcpGood())
            {
                /* Writing */
                Debug.Log("waiting to write");
                await WaitUntilAsync(DataReadyToSend);

                string json_str_data_to_send = JsonConvert.SerializeObject(obsv.Value);

                try
                {
                    if (SimulationManager._instance.debug_config.save_sent_packets)
                    {
                        await File.WriteAllTextAsync(SimulationManager._instance.debug_config.packet_sent_dir + "sent_data_" + observations_sent.ToString() + ".json", json_str_data_to_send);
                    }

                    Debug.Log("Sending: " + json_str_data_to_send);
                    byte[] _data = Encoding.UTF8.GetBytes(json_str_data_to_send);

                    if(udp_client != null)
                    {
                        await udp_client.SendAsync(_data, _data.Length);
                    }
                    else
                    {
                        await stream.WriteAsync(_data, 0, _data.Length);
                    }

                    last_obsv = obsv;
                    obsv = null;
                }

                catch (Exception e)
                {
                    Debug.LogException(e);
                }

                /* Reading */
                string jsonStr = "";

                if (udp_client != null)
                {
                    Debug.Log("I AM WAITING");
                    UdpReceiveResult result = await udp_client.ReceiveAsync();
                    jsonStr = Encoding.ASCII.GetString(result.Buffer, 0, result.Buffer.Length);
                    IPEndPoint sender = result.RemoteEndPoint;
                }
                else
                {
                    int bytes_read = await stream.ReadAsync(receive_buffer, 0, receive_buffer.Length);

                    if (bytes_read == 0)
                    {
                        await Task.Yield();
                        break;
                    }

                    jsonStr = Encoding.Default.GetString(receive_buffer);
                    Array.Clear(receive_buffer, 0, receive_buffer.Length);
                }

                if (jsonStr != null)
                {
                    Debug.Log("Received: " + jsonStr);
                    JsonMessage message = JsonConvert.DeserializeObject<JsonMessage>(jsonStr);

                    try
                    {
                        switch (message.msgType)
                        {
                            case "process_server_config":
                                json_server_config = message;
                                json_server_config.is_overriden = true;
                                break;
                            case "reset_episode":
                                json_reset_episode = message;
                                json_reset_episode.is_overriden = true;
                                break;
                            case "receive_json_controls":
                                json_rover_controls = message;
                                json_rover_controls.is_overriden = true;
                                break;
                            case "end_simulation":
                                json_end_simulation = message;
                                json_end_simulation.is_overriden = true;
                                break;
                            default:
                                break;
                        }
                    }

                    catch (Exception e)
                    {
                        Debug.LogException(e);
                    }
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
}
