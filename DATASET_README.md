# UAV-based wireless multi-modal measurements from AERPAW autonomous data mule (AADM) challenge in digital twin and real-world environments

Dataset DOI: [10.5061/dryad.7d7wm3898](https://doi.org/10.5061/dryad.7d7wm3898)

## Description of the data and file structure

The data were collected as part of the AERPAW Autonomous Aerial Data Mule (AADM) Challenge, a multi-institution effort to evaluate how autonomous UAVs can gather wireless data from distributed ground base stations. Participating teams designed flight control and decision-making algorithms that enabled a UAV to decide which base station to communicate with, how long to remain connected, and how to plan its trajectory to maximize total data download within a 500-second mission. Experiments were conducted in two phases: (1) a high-fidelity digital twin environment used for algorithm development and repeatable testing, and (2) real-world outdoor flights at the AERPAW Lake Wheeler Field Labs. During each mission, the UAV and ground nodes recorded synchronized telemetry, link-quality measurements, and cumulative throughput, while external instruments - including Keysight RF sensors, LoRa gateways, and a Fortem R20 radar - independently monitored the UAV to enrich the dataset with localization and sensing observations.

### Files and variables

#### File: AADM2025Dryad.zip

**Description:** The dataset is organized into three primary directories:

1. **LoRa**
2. **RF Sensor and Radar** 
3. **USRP**

Each directory contains measurements from the 11 finalist teams in the AERPAW Autonomous Aerial Data Mule (AADM) competition, with 3 real-world flights per team (33 total testbed flights). The USRP directory additionally includes development and preliminary test flights. The RF Sensor and Radar folder also has an additional three flights for localization and UAV tracking data.

#### 1. LoRa Directory

**Folder Organization**

* 33 subfolders - one per testbed flight (11 teams x 3 flights)
  * LORAlog.csv - Primary Data File
  * date_time_vehicleOut.txt -  ground-truth UAV telemetry
* Additional files:
  * All_Logs.csv - consolidated LoRa data from all flights
  * extract_packets_vars.m and extract_packets_vars_usage.m - MATLAB scripts for parsing and visualization
  * separate_vehicle_logs/ - folder including scripts to align LoRa data with UAV ground-truth telemetry
  * post-processing/ - folder including scripts and functions to post-process LoRa data, and regenerate figures included in subfolders of post-processing
  * AADM_Collections_Fig_11/ - folder including data and scripts to reproduce figures from our IEEE Data Descriptions paper about this dataset. A README is included in this folder for more information
  * **README.md - Please make sure to refer here for more information on post-processing instructions and dataset notes**

**Primary Data File**

* LORAlog.csv (one per flight)

**Key LoRa Variables**

| Variable                              | Description                                                                                            |
| :------------------------------------ | :----------------------------------------------------------------------------------------------------- |
| deduplication\_id                     | Unique identifier assigned to a LoRa packet to remove duplicate packets received by multiple gateways. |
| time                                  | Timestamp indicating when the LoRa packet was received by the network server. |
| dev\_eui                              | Unique identifier for the LoRa end device (UAV transmitter). |
| dr                                    | LoRa data rate used for the transmission. |
| f\_cnt                                | Frame counter used in LoRaWAN to track packet sequence and detect missing transmissions. |
| data                                  | Encoded payload transmitted by the LoRa device containing the application data. |
| rx\_info\_snr                         | Signal-to-Noise Ratio (SNR) measured at the gateway when the packet was received. |
| rx\_info\_rssi                        | Received Signal Strength Indicator (RSSI) measured at the gateway, indicating received signal power. |
| rx\_info\_time                        | Timestamp recorded by the gateway at the moment the packet was received. |
| rx\_info\_channel                     | Radio channel index used by the gateway to receive the LoRa packet. |
| rx\_info\_rfChain                     | RF receiver chain on the gateway that processed the signal. |
| rx\_info\_latitude                    | Latitude coordinate of the receiving gateway. |
| rx\_info\_longitude                   | Longitude coordinate of the receiving gateway. |
| rx\_info\_altitude                    | Altitude (m) of the gateway above sea level. |
| rx\_info\_region\_name                | Name of the LoRaWAN regional configuration used for the transmission. |
| rx\_info\_region\_common\_name        | Common regional band designation for the LoRa network. |
| rx\_info\_uplinkId                    | Unique identifier assigned to the uplink message instance received by the gateway.                     |
| rx\_info\_gatewayId                   | Unique identifier of the gateway that received the packet. |
| rx\_info\_timeSinceGpsEpoch           | Time elapsed since the GPS epoch used for precise time synchronization. |
| tx\_info\_frequency                   | Radio frequency used for the LoRa transmission. |
| tx\_info\_modulation\_bandwidth       | Bandwidth of the LoRa modulation used during transmission. |
| tx\_info\_modulation\_spreadingFactor | Spreading factor used for the LoRa transmission, which affects range and data rate. |
| tx\_info\_modulation\_codeRate        | Forward error correction coding rate used in the LoRa modulation. |

Purpose: Characterizes long-range low-power uplink performance between the UAV and distributed LoRa gateways.

#### 2. RF Sensor and Radar Directory

**Folder Organization**

* 33 subfolders - one per testbed flight with all modalities
* 3 additional folders (OptX) - flights with radar + RF only (localization and tracking experiments)
* Post_Processing_Scripts/ - folder with MATLAB/Jupyter scripts for general post-processing of data. The folder also includes a file named AADM_Collections_Figure_10.m and the corresponding radar (.txt) and RF sensor data (.csv) to recreate Figure 10 from our IEEE Data Descriptions paper.
* README.md - contains comprehensive instructions for executing the scripts in the Post_Processing_Scripts folder

**Files per Flight**

1. *AADMx.csv* - Keysight RF sensor localization outputs
2. *radar_data_N.json* -  Fortem R20 radar tracking logs
3. *date*_*time*_vehicleOut.txt -  ground-truth UAV telemetry

\**The **Opt1** folder contains the three aforementioned files, as well as an additional set of three files labeled **rerun**. During the first run of the experiment, the UAV transmitted a channel sounding waveform at a lower bandwidth than intended, which resulted in reduced RF sensor localization accuracy. The **rerun** files were collected with the correct transmitted bandwidth and therefore provide higher localization accuracy. These rerun measurements are consistent with the bandwidth used in the other optimized flights in **Opt2** and **Opt3**.**

**Radar Variables (JSON)**

| Variable       | Description                          |
| :------------- | :----------------------------------- |
| id             | Track identification number          |
| range          | Target range (m)                     |
| azimuth        | Azimuth angle (deg)                  |
| elevation      | Elevation angle (deg)                |
| rcsDbsm        | Radar cross section (dBsm)           |
| radialVelocity | Radial velocity (m/s, + = receding)  |
| lla            | Latitude/longitude/altitude of track |
| gpsX           | GPS time at CPI center               |

**RF Sensor Variables (CSV)**

| Variable           | Description                                                                   |
| :----------------- | :---------------------------------------------------------------------------- |
| Meas ID            | Unique identifier assigned to each measurement record in the flight. |
| Algorithm          | Localization algorithm used to compute the measurement (e.g., Keysight TDOA). |
| Center Frequency   | Center frequency of localized UAV signal (Hz). |
| Sample Rate        | Sampling rate of the recorded RF signal in samples per second (Hz). |
| Num Samples        | Number of signal samples recorded for the measurement. |
| Latitude           | Latitude of UAV position estimate (deg). |
| Longitude          | Longitude of UAV position estimate (deg). |
| RHO                | Degree of cross-correlation                                                   |
| CEP                | Circular error probability (m)                                                |
| Total Sensors      | Total number of RF sensors used in the measurement. |
| Valid Sensors      | Number of sensors that provided usable data for the measurement. |
| Overloaded Sensors | Number of sensors that experienced signal overload during the measurement. |
| Sensor Names       | Identifiers (IP addresses) of the sensors participating in the measurement. |
| Time               | Timestamp of position estimate. |

Purpose: Independent UAV localization via passive RF sensing and active radar tracking.

#### 3. USRP Directory (Core AADM Measurements)

**Subdirectories**

* Development 45 flights Jun/ - 45 digital-twin flights
* Testbed 2 flights Mar and April/ - 2 preparation flights, USRP only
* Testbed 2 flights May ACW Demo/ -  2 preparation flights, USRP only
* Testbed 8 Flights Jun/ - 8 preparation flights, USRP only
* Testbed 33 flights Sep/ - 33 official competition flights, all modalities
* Post-Processing/ - folder with Matlab scripts for post-processing and visualization of data
* README.md - Instructions for running MATLAB scripts and post processing data

**Structure per Flight**

Each flight folder contains:

* *LW1/, LW2/, LW3/, LW4/ -* data from Base Stations 1-4
* *UAV/* - onboard telemetry

Files in LW#  folders:

* *controller_log.txt* - recorded messages from experiment control software

- *power_log.txt* - received signal power
- *quality_log.txt* - link quality metrics
- *radio_channelsounderrxgrc_log.txt -* recorded messages from the GNU Radio channel sounder receiver
- *snr_log.txt* - per-second SNR reported to controller
- *responder_log.txt* - BS response messages

Files in UAV folder:

* *radio_channelsoundertxgrc_log.txt* - Log file generated by the GNU Radio channel sounder transmitter

- *report_log.txt* - UAV and controller exchanges
- *vehicle_log.txt -* recording of experiment control events including waypoint timing, base station SNR measurements, and data download decisions
- *vehicleOut.txt* -  ground-truth UAV telemetry

Processed CSVs:

* *vehicleOut_snr.csv*
* *vehicleOut_down.csv*

**UAV Telemetry and Link Variables**

| Variable            | Description                       |
| :------------------ | :-------------------------------- |
| time                | Timestamp (\~1 Hz)                |
| Altitude            | UAV altitude (m)                  |
| BatteryVolts        | Battery voltage (V)               |
| GPSFix              | GNSS fix indicator                |
| Latitude, Longitude | UAV position (deg)                |
| NumberOfSatellites  | GNSS Satellites                   |
| Pitch, Roll, Yaw    | Orientation (rad)                 |
| VelocityX/Y/Z       | Body-frame velocity (m/s)         |
| dl\_lwN             | Cumulative data from BS N (Mbits) |
| snr\_lwN            | SNR to BS N (dB)                  |

**Controller Variables**

| Variable             | Description           |
| :------------------- | :-------------------- |
| assigned\_data       | Data volume per BS    |
| selected\_BS         | Current serving BS    |
| inst\_rate           | Estimated data rate   |
| cumulative\_download | Mission progress      |
| download\_enabled    | On/off flag           |
| mission\_time        | Elapsed time (<500 s) |

**Dataset Coverage Summary**

| Component           | # Flights | Modalities               |
| :------------------ | :-------- | :----------------------- |
| Development DT      | 45        | USRP only                |
| Early Testbed       | 12        | USRP                     |
| Competition Testbed | 33        | USRP + LoRA + RF + Radar |
| Localization Extras | 3         | Radar + RF               |

## Code/software

All data processing and analysis are designed to be performed in MATLAB, with the exception of a single Jupyter notebook provided to convert the Fortem radar *.json files into MATLAB-compatible .mat format. Post-processing scripts are included within each of the three modality subdirectories (USRP, LoRa, and RF Sensor and Radar) and support data loading, time alignment, and generation of representative figures. The MATLAB workflows rely only on standard base functionality, with the Mapping Toolbox required for the radar and RF sensor processing scripts to enable geospatial coordinate transformations and visualizations. The provided code is intended as reference implementations for parsing raw logs, extracting key variables, and reproducing the example analyses included with the dataset. Users may execute the radar conversion notebook once to generate .mat files and then complete all subsequent analysis entirely within MATLAB.

## Access information

Other publicly accessible locations of the data:

* NA

Data was derived from the following sources:

* NA

