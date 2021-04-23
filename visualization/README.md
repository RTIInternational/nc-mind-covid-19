## Creating Visualization Data

The script in this folder is used record the movements of people in a standard model run for the purpose of visualization. The actual visualization code is currently not a part of this repository.

To generate a visualization data file, run the python file `record_model_for_visualization.py`.

Arguments:
  * *(optional, default ../covid19-reporting/animated_visualization/data)*`--output_folder`: This is the output location of the visualization data file. Note that the default assumes the visualization repository is on the same folder level as this repository.
  * *(optional, default 1.2)*`--R0`: This float controls the strength of covid in the model. Higher R0's lead to more agents being infected each day.
  * *(optional, default None)*`--county_fips`: This argument can be used to subset the model run by people in a particular county
  * *(optional, default 10000)*`--limit_pop`: This argument sets the model population to use in the run. Higher values are more representative but may have lag when rendering in the visualization.
  * *(optional, default 0.0)*`--ratio_to_hospital`: This is the portion of mild to moderate covid cases which seek hospitalization.
  * *(optional, default 0.79)*`--community_probability_multiplier`: This is a multiplier which controls how often people leave the community.


Output:
  * `data/vis_data.json`: a json file with a dictionary for each agent recorded during the model run. This dictionary includes the following information.
    * For each Agent in the ABM which moves from the community at some point
       * `ID`: The unique identifier for the agent
       * `covid_status`: The integer values for the covid status of the agent at each timestep.
       * `loc_cats`: The string location categories for each location of the agent at each timestep.
       * `alive`: A binary list of the agent's alive/dead status for each timestep.
   * For each agent in the ABM which never leaves the community:
       * `ID`: The unique identifier for the agent
       * `covid_status`: The integer values for the covid status of the agent at each timestep.
       * `alive`: A binary list of the agent's alive/dead status for each timestep.
   * Bed counts for each facility category
   * The arguments used to generate the data
