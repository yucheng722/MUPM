"""Download data coding for selected fields.

For any field, e.g. field 31 for sex, the information can be found in the page:
https://biobank.ndph.ox.ac.uk/showcase/field.cgi?id=31

For any data coding, e.g. data coding 9 for sex, the information can be found in the page:
https://biobank.ndph.ox.ac.uk/showcase/coding.cgi?id=9
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests
from torch.ao.quantization import default_weight_observer
from tqdm import tqdm
from datetime import datetime
import ast
import json
import random
import pdb

def combine_strings(strings):
    if not strings:
        return ""
    elif len(strings) == 1:
        return strings[0]
    elif len(strings) == 2:
        return f"{strings[0]} and {strings[1]}"
    else:
        return f"{', '.join(strings[:-1])} and {strings[-1]}"

def download_coding_files(coding_id: int, out_dir: Path, overwrite: bool = False) -> Path | None:
    """Download data-coding files using a POST request and save to the specified directory.

    Args:
        coding_id: The ID of the file to download.
        out_dir: The directory to save the downloaded file.
        overwrite: Whether to overwrite the existing file.

    Returns:
        Output file path if successful, otherwise None.
    """
    try:
        file_path = Path(out_dir / f"coding{coding_id}.tsv")
        if file_path.exists() and not overwrite:
            #logger.info(f"File {file_path} already exists.")
            return file_path

        response = requests.post(
            "https://biobank.ndph.ox.ac.uk/showcase/codown.cgi", data={"id": coding_id}, stream=True, timeout=10
        )
        response.raise_for_status()
        with file_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        #logger.info(f"Downloaded data for {coding_id} to {file_path}.")

    except requests.RequestException:
        #logger.exception("Error downloading file")
        return None
    return file_path


def read_csv_by_chunk(file_path: Path, columns: list[str], chunksize: int) -> pd.DataFrame:
    """Read a large CSV file in chunks and return a filtered DataFrame.

    Args:
        file_path: Path to the CSV file.
        columns: list of columns to read.
        chunksize: Number of rows to read per chunk.

    Returns:
        Filtered DataFrame with non-null rows.
    """
    usecols = ["eid", *columns, "20208-2.0", "20208-3.0", "20209-2.0", "20209-3.0"]
    sele_dfs = []
    for chunk_df in tqdm(pd.read_csv(file_path, usecols=usecols, low_memory=False, iterator=True, chunksize=chunksize)):
        # select only rows with both LAX/SAX views, either instance 2 or 3
        # and alter the eid to include the instance number
        instance_2_df = chunk_df.dropna(subset=["20208-2.0", "20209-2.0"], how="any", axis=0).copy()
        instance_2_df["eid"] = instance_2_df["eid"].astype(str) + "_2_0"
        instance_3_df = chunk_df.dropna(subset=["20208-3.0", "20209-3.0"], how="any", axis=0).copy()
        instance_3_df["eid"] = instance_3_df["eid"].astype(str) + "_3_0"
        sele_df = pd.concat([instance_2_df, instance_3_df])

        # drop rows with all NaN values
        sele_df = sele_df.dropna(subset=columns, how="all", axis=0)
        if sele_df.shape[0] > 0:
            sele_dfs.append(sele_df)
    # drop columns with all NaN values
    return pd.concat(sele_dfs).dropna(axis=1, how="all")


def get_field_to_map(field_df: pd.DataFrame, out_dir: Path) -> dict[int, dict[int, str]]:
    """Get mapping of field ID to data coding.

    Args:
        field_df: DataFrame with field information.
        out_dir: Folder to save the downloaded files.

    Returns:
        Mapping of field ID to data coding.
    """
    # download data coding files for all fields
    data_coding_ids = sorted(field_df["Data Coding"].dropna().astype(int).unique())
    data_coding_to_map = {}
    for x in data_coding_ids:
        file_path = download_coding_files(x, out_dir)
        if file_path is not None:
            data_coding_df = pd.read_csv(str(file_path), sep="\t", usecols=["coding", "meaning"])
            # 9: {0: 'Female', 1: 'Male'}
            data_coding_to_map[x] = data_coding_df.set_index("coding")["meaning"].to_dict()
    # {31: 9}, field 31 has data coding 9
    field_to_data_coding = field_df.dropna(subset=["Data Coding"]).set_index("UDI")["Data Coding"].to_dict()
    # {31: {0: 'Female', 1: 'Male'}}, field 31 has value 0 for female and 1 for male
    field_to_map = {int(k): data_coding_to_map[int(v)] for k, v in field_to_data_coding.items()}
    return field_to_map


def get_field_columns(row: pd.Series, field_id: int, instance: str) -> list[str]:
    """Get columns from DataFrame based on field ID and optional instance index.

    Args:
        row: Row of the patient DataFrame.
        field_id: Field ID to filter columns.
        instance: "2" for first image visit, "3" for second, "" means no filtering.

    Returns:
        list of column names matching the criteria.
    """
    columns = [x for x in row.index if x.split("-")[0] == str(field_id)]
    if instance == "":
        return columns
    # 20002-3.0 is the column for field 20002 for instance 3
    return [x for x in columns if x.split("-")[1].split(".")[0] == instance]


class Parser:
    """Parser to extract information from the metadata DataFrame."""

    def __init__(self, field_to_map: dict[int, dict[int, str]]) -> None:
        """Initialize the parser.

        Args:
            field_to_map: Mapping of field ID to data coding.
        """
        self.field_to_map = field_to_map

        cardiac_dx_df = pd.read_csv(Path(__file__).parent.resolve() / "cardiac_diagnosis_ids.csv")
        self.cardiac_dx_include_values = tuple(i[1]["ids"] for i in cardiac_dx_df.iterrows())

    def get_values_with_map(
        self,
        row: pd.Series,
        field_id: int,
        instance: str,
        include_values: tuple | None = None,  # type: ignore[type-arg]
        exclude_values: tuple | None = None,  # type: ignore[type-arg]
        include_keyword: str = "",
    ) -> list:  # type: ignore[type-arg]
        """Get values corresponding to the field and map to text if data coding is available.

        Args:
            row: Row of the patient DataFrame.
            field_id: Field ID to filter columns.
            instance: "2" for first image visit, "3" for second.
            include_values: Tuple of values to include.
            exclude_values: Tuple of values to exclude.
            include_keyword: Keyword to include in the values.

        Returns:
            list of meanings for the filtered values.
        """
        # only one of three filters can be applied
        if sum([include_values is not None, exclude_values is not None, include_keyword != ""]) > 1:
            raise ValueError("Only one of include_values, exclude_values, include_keyword can be used.")

        columns = get_field_columns(row, field_id, instance)
        values = row[columns].dropna().tolist()
        if field_id in self.field_to_map:
            coding_map = self.field_to_map[field_id]
            if include_values:
                values = [coding_map[x] for x in values if x in include_values]
            elif exclude_values:
                values = [coding_map[x] for x in values if x not in exclude_values]
            elif include_keyword:
                values = [coding_map[x] for x in values if include_keyword in str(x)]
            else:
                values = [coding_map[x] for x in values]
        return values

    def get_age(self, row: pd.Series, instance: str) -> str:
        """Calculate the age of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Calculated age.
        """
        attending_year = row[f"53-{instance}.0"].split("-")[0]  # 2018-05-22 -> 2018
        birth_year = row["34-0.0"]
        return str(int(attending_year) - int(birth_year))

    def get_sex(self, row: pd.Series) -> str:
        """Get sex of the patient.

        Args:
            row: Row of the patient DataFrame.

        Returns:
            Male or Female.
        """
        return self.get_values_with_map(row, field_id=31, instance="")[0]

    def get_weight(self, row: pd.Series, instance: str) -> str:
        """Get weight of the patient, unit in kg.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Weight.
        """
        return f'{row[f"12143-{instance}.0"]:.1f}'

    def get_height(self, row: pd.Series, instance: str) -> str:
        """Get height of the patient, unit in cm.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Height.
        """
        return f'{row[f"12144-{instance}.0"]:.1f}'

    def get_BMI(self, row: pd.Series, instance: str) -> str:
        """Get BMI of the patient, unit in Kg/m2.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            BMI.
        """
        weight = float(self.get_weight(row, instance))
        height = float(self.get_height(row, instance))
        BMI_score = weight / (height/100) * (height/100)

        return f'{BMI_score:.1f}'

    def get_symptoms(self, row: pd.Series, instance: str) -> str:
        """Get symptoms of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            symptoms.
        """
        #chest_pain = row[f"2335-{instance}.0"]
        chest_pain = row[f"2335-{instance}.0"]
        wheeze_or_whistling = row[f"2316-{instance}.0"]
        shortness_of_breath = row[f"4717-{instance}.0"]
        symptoms = []
        # -1 represents "Do not know"
        # -3 represents "Prefer not to answer"

        if chest_pain > 0:
            chest_pain_walking = row[f"3606-{instance}.0"]
            chest_pain_walking_uphill = row[f"3751-{instance}.0"]
            chest_pain_standing_still = row[f"3616-{instance}.0"]

            if chest_pain_walking > 0:
                if chest_pain_walking_uphill > 0:
                    if chest_pain_standing_still > 0:
                        symptoms.append("chest pain that occurs during walking uphill or hurrying and resolves when standing still")
                    else:
                        symptoms.append("chest pain that occurs during walking uphill or hurrying")
                else:
                    symptoms.append("chest pain that occurs during walking")
            else:
                symptoms.append("chest pain")
        elif chest_pain == 0:
            symptoms.append("no chest pain")
        
        if wheeze_or_whistling > 0:
            symptoms.append("wheezing or a whistling sound in the chest")
        elif wheeze_or_whistling == 0:
            symptoms.append("no wheezing or a whistling sound in the chest")

        if shortness_of_breath > 0:
            symptoms.append("shortness of breath while walking on level ground")
        elif shortness_of_breath == 0:
            symptoms.append("no shortness of breath while walking on level ground")

        return combine_strings(symptoms)

    def get_pulse_rate(self, row: pd.Series, instance: str) -> str:
        """Get pulse rate of the patient, unit in beats per minute.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            pulse rate.
        """
        pulse_rate = row[f"102-{instance}.0"]
        if pd.isna(pulse_rate):
            return None
        else: 
            return f'{pulse_rate:.0f}'

    def get_diastolic_blood_pressure(self, row: pd.Series, instance: str) -> str:
        """Get diastolic blood pressure of the patient, unit in mmHg.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            diastolic blood pressure.
        """
        diastolic_blood_pressure = row[f"4079-{instance}.0"]
        if pd.isna(diastolic_blood_pressure):
            return None
        else: 
            return f'{diastolic_blood_pressure:.0f}'

    def get_systolic_blood_pressure(self, row: pd.Series, instance: str) -> str:
        """Get systolic blood pressure of the patient, unit in mmHg.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            systolic blood pressure.
        """
        systolic_blood_pressure = row[f"4080-{instance}.0"]
        if pd.isna(systolic_blood_pressure):
            return None
        else: 
            return f'{systolic_blood_pressure:.0f}'


    def calculate_year(self, row: pd.Series, instance: str, query_time: str, analysis_mode: str) -> str:
        """Calculate the number of years between the query time and the time the MRI was taken.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            the number of years.
        """
        mri_time = pd.to_datetime(row[f"53-{instance}.0"])
        query_time = pd.to_datetime(query_time)
        
        if analysis_mode == "history":
            time_diff = (mri_time - query_time).days
        elif analysis_mode == "prediction":
            time_diff = (query_time - mri_time).days
        
        if time_diff < 0:
            return None
        else:
            return time_diff
        # elif 0 < time_diff < 1:
        #     return f"{int(abs((mri_time - query_time).days / 30))} months"
        # elif time_diff > 1:
        #     return f"{int(time_diff)} years"

    def clean_string(self, phrases_to_remove, input_string):
        # Make sure input_string is initialized before using it
        if input_string is None:
            return ""
        
        # Convert input_string to string if it isn't already
        input_string = str(input_string)
        
        # Remove each phrase
        for phrase in phrases_to_remove:
            input_string = re.sub(phrase, "", input_string, count=1).strip()
        
        return input_string

    def get_diseases_with_time(
            self, 
            row: pd.Series, 
            instance: str, 
            analysis_mode: str,
            phrases_to_remove: list | None = None,
            include_values: list | None = None, 
            customed_category: dict | None = None) -> dict:
        """Get cardaic history of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            cardaic history.
        """

        if (include_values is None and customed_category is None) or (include_values is not None and customed_category is not None):
            raise ValueError("Exactly one of 'include_values' or 'customed_category' must be provided, but not both.")

        disease_columns = get_field_columns(row, field_id=41270, instance="")
        time_columns = get_field_columns(row, field_id=41280, instance="")
        diseases = row[disease_columns].dropna().tolist()
        times = row[time_columns].dropna().tolist()
        coding_map = self.field_to_map[41270]

        diseases_with_time = {}
        if include_values is not None:
            for disease_code, time in zip(diseases, times):
                disease_code = disease_code[0:3]
                if disease_code in include_values:
                    disease_name = self.clean_string(phrases_to_remove, coding_map[disease_code][4:].lower())
                    if disease_name not in diseases_with_time or time < diseases_with_time[disease_name]:
                        diseases_with_time[disease_name] = time        
        elif customed_category is not None:
            code_to_category = {}
            for category, codes in customed_category.items():
                for code in codes:
                    code_to_category[code] = category
            for disease_code, time in zip(diseases, times):
                disease_code = disease_code[0:3]
                if disease_code in code_to_category:
                    category = code_to_category[disease_code]
                    if category not in diseases_with_time or time < diseases_with_time[category]:
                        diseases_with_time[category] = time

        analyzed_result = {}
        for disease_name, time in diseases_with_time.items():
            time_diff = self.calculate_year(row, instance, time, analysis_mode)
            if time_diff:
                analyzed_result[disease_name] = time_diff

        return analyzed_result

    def get_cardiac_related_diseases_with_time(self, row: pd.Series, instance: str, analysis_mode: str) -> str:

        phrases_to_remove = [
            ", not specified as haemorrhage or infarction",
            ", not resulting in cerebral infarction",
            "in diseases classified elsewhere",
            "other specified",
            "other",
        ]
        include_values = [f"I{str(i).zfill(2)}" for i in range(60, 80)] + [f"E{str(i).zfill(2)}" for i in range(10, 15)] + [f"E{str(i).zfill(2)}" for i in range(0, 8)] + ['E78', 'N18', 'G47']
        return self.get_diseases_with_time(row, instance=instance, include_values=include_values, phrases_to_remove=phrases_to_remove, analysis_mode=analysis_mode)

    def get_cardiac_diseases_with_time(self, row: pd.Series, instance: str, analysis_mode: str) -> str:

        phrases_to_remove = [
            "in diseases classified elsewhere",
            "and ill-defined descriptions",
            "other"
        ]
        include_values = [f"I{str(i).zfill(2)}" for i in range(0, 53)] + [f"Q{str(i).zfill(2)}" for i in range(20, 25)]
        return self.get_diseases_with_time(row, instance=instance, include_values=include_values, phrases_to_remove=phrases_to_remove, analysis_mode=analysis_mode)

    def get_customed_category_with_time(self, row: pd.Series, instance: str, analysis_mode: str) -> str:

        customed_category = {
            "hypertensive disease": ['I10', 'I11', 'I12', 'I13', 'I14', 'I15'],
            "ischaemic heart disease": ['I20', 'I21', 'I22', 'I23', 'I24', 'I25'],
            "cardiac arrhythmia": ['I46', 'I47', 'I48', 'I49'],
            "conduction disorder": ['I44', 'I45'],
            "complication heart disease": ['I51'],
            "valve disorder": ['I34', 'I35', 'I36', 'I37', 'I38', 'I39'],
            "heart failure": ['I50'],
            "pulmonary heart disease": ['I26', 'I27', 'I28'],
            "chronic rheumatic heart disease": ['I05', 'I06', 'I07', 'I08', 'I09'],
            "pericarditis": ['I30', 'I31', 'I32'],
            "cardiomyopathy": ['I42', 'I43']
        }
        # customed_category = {
        #     "A. Hypertensive Disease": ['I10', 'I11', 'I12', 'I13', 'I14', 'I15'],
        #     "B. Ischaemic Heart Disease": ['I20', 'I21', 'I22', 'I23', 'I24', 'I25'],
        #     "C. Cardiac Arrhythmia": ['I46', 'I47', 'I48', 'I49'],
        #     "D. Conduction Disorder": ['I44', 'I45'],
        #     "E. Complication Heart Disease": ['I51'],
        #     "F. Valve Disorder": ['I34', 'I35', 'I36', 'I37', 'I38', 'I39'],
        #     "G. Heart Failure": ['I50'],
        #     "H. Pulmonary Heart Disease": ['I26', 'I27', 'I28'],
        #     "I. Chronic Rheumatic Heart Disease": ['I05', 'I06', 'I07', 'I08', 'I09'],
        #     "J. Pericarditis": ['I30', 'I31', 'I32'],
        #     "K. Cardiomyopathy": ['I42', 'I43']
        # }
        return self.get_diseases_with_time(row, instance=instance, customed_category=customed_category, analysis_mode=analysis_mode)

    def get_family_history(self, row: pd.Series, instance: str) -> str:
        """Get family history information for the patient.

        Args:
            row: Row of the patient DataFrame.

        Returns:
            Family history information summary.
        """
        include_values = [1.0, 2.0, 8.0]
        father_medical_history = self.get_values_with_map(row, field_id=20107, instance=instance, include_values=include_values)
        mother_medical_history = self.get_values_with_map(row, field_id=20110, instance=instance, include_values=include_values)
        siblings_medical_history = self.get_values_with_map(row, field_id=20111, instance=instance, include_values=include_values)
        medical_histories = []
        if father_medical_history:
            medical_histories.append(f"the patient's father having a history of {combine_strings(father_medical_history).lower()}")
        if mother_medical_history:
            medical_histories.append(f"the patient's mother having a history of {combine_strings(mother_medical_history).lower()}")
        if siblings_medical_history:
            medical_histories.append(f"patient's siblings having a history of {combine_strings(siblings_medical_history).lower()}")
        return combine_strings(medical_histories)+'.'

    def get_MET_score(self, row: pd.Series, instance: str) -> str:
        """Get MET score of the patient, unit in minutes.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            MET score.
        """
        moderate_activity = row[f"22038-{instance}.0"]
        vigorous_activity = row[f"22039-{instance}.0"]
        walking = row[f"22037-{instance}.0"]

        if pd.isna(moderate_activity) or pd.isna(vigorous_activity) or pd.isna(walking):
            return None
        else: 
            return moderate_activity+vigorous_activity+walking

    def get_sleep_duration(self, row: pd.Series, instance: str) -> str:
        """Get sleep duration of the patient, unit in hours.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            sleep duration.
        """
        sleep_duration = row[f"1160-{instance}.0"]
        if sleep_duration>0:
            return sleep_duration
        else:
            return None

    def get_implants_and_grafts(self, row: pd.Series) -> str:
        """Get information about the presence of cardiac and vascular implants and grafts.

        41270 is a summary of the distinct diagnosis codes a participant has had recorded across all their hospital
        inpatient records in either the primary or secondary position.

        We are interested in the following family history fields:

            Z95 Presence of cardiac and vascular implants and grafts
            Z95.0 Presence of cardiac pacemaker
            Z95.1 Presence of aortocoronary bypass graft
            Z95.2 Presence of prosthetic heart valve
            Z95.3 Presence of xenogenic heart valve
            Z95.4 Presence of other heart-valve replacement
            Z95.5 Presence of coronary angioplasty implant and graft
            Z95.8 Presence of other cardiac and vascular implants and grafts
            Z95.9 Presence of cardiac and vascular implant and graft, unspecified

        Args:
            row: Row of the patient DataFrame.

        Returns:
            Cardiac operation history information summary.
        """
        include_values = ("Z95", "Z950", "Z951", "Z952", "Z953", "Z954", "Z955", "Z958", "Z959")
        history = self.get_values_with_map(row, field_id=41270, instance="", include_values=include_values)
        pattern = re.compile(r"Presence of (.+)")
        return ", ".join([match.group(1) for x in history if (match := pattern.search(x))])

    def get_smoking_report(self, row: pd.Series, instance: str) -> str:  # noqa: C901, pylint:disable=too-many-branches
        """Get smoking status of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Smoking status.
        """
        # -3, prefer not to answer; 0, never; 1, previous; 2, current
        status = row[f"20116-{instance}.0"]
        if status == -3:
            return ""
        if status == 0:
            return "never smoker"
        if status == 2:
            report = "current smoker"

            # -1 represents "Do not know"
            # -3 represents "Prefer not to answer"
            age_started = row[f"3436-{instance}.0"]
            if age_started > 0:
                report += f", started at age {int(age_started)}"

            # -10 represents "Less than one a day"
            # -1 represents "Do not know"
            # -3 represents "Prefer not to answer"
            n_cigarettes = row[f"3456-{instance}.0"]
            frequency_prefix = ""
            if n_cigarettes == -10:
                report += ", smokes less than one cigarette per day"
            elif n_cigarettes > 0:
                report += f", smokes {int(n_cigarettes)} cigarettes per day"
            else:
                frequency_prefix = " smokes"

            # 1	Yes, on most or all days
            # 2	Only occasionally
            frequency = row[f"1239-{instance}.0"]
            if frequency == 2:
                report += f",{frequency_prefix} occasionally"
            elif frequency == 1:
                report += f",{frequency_prefix} on most or all days"

            return report

        report = "previous smoker"

        # -1 represents "Do not know"
        # -3 represents "Prefer not to answer"
        age_started = row[f"2867-{instance}.0"]
        if age_started > 0:
            report += f", started at age {int(age_started)}"

        # -10 represents "Less than one a day"
        # -1 represents "Do not know"
        n_cigarettes = row[f"2887-{instance}.0"]
        frequency_prefix = ""
        if n_cigarettes == -10:
            report += ", smoked less than one cigarette per day"
        elif n_cigarettes > 0:
            report += f", smoked {int(n_cigarettes)} cigarettes per day"
        else:
            frequency_prefix = " smoked"

        # 1	Smoked on most or all days
        # 2	Smoked occasionally
        # 3	Just tried once or twice
        # 4	I have never smoked
        # -3 Prefer not to answer
        frequency = row[f"1249-{instance}.0"]
        if frequency == 1:
            report += f",{frequency_prefix} on most or all days"
        elif frequency == 2:
            report += f",{frequency_prefix} occasionally"
        elif frequency == 3:
            report += ", tried once or twice"

        # -1 represents "Do not know"
        # -3 represents "Prefer not to answer"
        age_stopped = row[f"2897-{instance}.0"]
        if age_stopped > 0:
            report += f", stopped at age {int(age_stopped)}"
        report += '.'
        return report

    def get_drinking_report(self, row: pd.Series, instance: str) -> str:
        """Get drinking status of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Drinking status.
        """
        # -3, prefer not to answer; 0, never; 1, previous; 2, current
        status = row[f"20117-{instance}.0"]
        if status == -3:
            return ""
        if status == 0:
            return "never drinker"
        if status == 1:
            report = "previous drinker"
            verb = "drank"
        elif status == 2:
            report = "current drinker"
            verb = "drinks"
        else:
            return ""

        # 6	Never
        # -3 Prefer not to answer
        frequency = self.get_values_with_map(row, field_id=1558, instance=instance, exclude_values=(6, -3))
        if frequency:
            if frequency[0] == "special occasions only":
                report += f", {verb} on special occasions only"
            else:
                # one to three times a month
                report += f", {verb} {frequency[0].lower()}"
        report += '.'
        return report

    def get_physical_activity(self, row: pd.Series, instance: str) -> str:
        """Get physical activity status of the patient.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Physical activity status.
        """
        # -1 represents "Do not know"
        # -3 represents "Prefer not to answer"
        n_days_moderate = row[f"884-{instance}.0"]
        n_days_vigorous = row[f"904-{instance}.0"]
        if n_days_moderate <= 0 and n_days_vigorous <= 0:
            return ""
        report = "Do "
        if n_days_moderate > 0:
            report += f"moderate physical activities on {int(n_days_moderate)} days per week"
        if n_days_vigorous > 0:
            if n_days_moderate > 0:
                report += " and "
            report += f"vigorous physical activities on {int(n_days_vigorous)} days per week"
        return report

    def get_ecg_diagnoses(self, row: pd.Series, instance: str) -> str:
        """Get ECG automated diagnoses.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            Cardiac operation history information summary.
        """
        reports = self.get_values_with_map(row, field_id=12653, instance=instance)
        reports = [x for x in reports if x != "---"]
        return "; ".join(reports)

    def get_heart_function(self, row: pd.Series, instance: str) -> str:
        """Get heart function.

        Args:
            row: Row of the patient DataFrame.
            instance: "2" for first image visit, "3" for second.

        Returns:
            heart function.
        """
        end_diastolic_volume = row[f"22421-{instance}.0"]
        end_systolic_volume = row[f"22422-{instance}.0"]
        ejection_fraction = row[f"22420-{instance}.0"]
        
        if not pd.isna(end_diastolic_volume):
            return f"The patient's left ventricular end-diastolic volume was {end_diastolic_volume} ml, end-systolic volume was {end_systolic_volume} ml, and ejection fraction was {ejection_fraction}%."
        else:
            return None

    def get_MRI_taken_time(self, row: pd.Series, instance: str) -> str:
        MRI_taken_time = row[f"53-{instance}.0"]
        if pd.isna(MRI_taken_time):
            return None
        else: 
            return MRI_taken_time

    def parse_row(self, row: pd.Series) -> dict[str, str]:
        """Parse a row of the DataFrame.

        Args:
            row: Row of the DataFrame.

        Returns:
            Dictionary of parsed information.
        """
        instance = row["eid"].split("_")[1]
        # TODO: PWA related fields
        # TODO: blood pressure related fields
        return {
            "eid": row["eid"],
            "age": self.get_age(row, instance=instance),
            "sex": self.get_sex(row),  # sex is not instance specific
            "weight": self.get_weight(row, instance=instance),
            "height": self.get_height(row, instance=instance),
            "BMI": self.get_BMI(row, instance=instance),
            "symptoms": self.get_symptoms(row, instance=instance),
            "pulse_rate": self.get_pulse_rate(row, instance=instance),
            "diastolic_blood_pressure": self.get_diastolic_blood_pressure(row, instance=instance),
            "systolic_blood_pressure": self.get_systolic_blood_pressure(row, instance=instance),
            "cardiac_related_history_with_time": self.get_cardiac_related_diseases_with_time(row, instance=instance, analysis_mode="history"),
            "cardiac_history_with_time": self.get_cardiac_diseases_with_time(row, instance=instance, analysis_mode="history"),
            "cardiac_related_prediction_with_time": self.get_cardiac_related_diseases_with_time(row, instance=instance, analysis_mode="prediction"),
            "cardiac_prediction_with_time": self.get_cardiac_diseases_with_time(row, instance=instance, analysis_mode="prediction"),
            "customed_category_prediction_with_time": self.get_customed_category_with_time(row, instance=instance, analysis_mode="prediction"),
            "family_history": self.get_family_history(row, instance=instance),
            "MET": self.get_MET_score(row, instance=instance),
            "sleep_duration": self.get_sleep_duration(row, instance=instance),
            "smoking": self.get_smoking_report(row, instance=instance),
            "drinking": self.get_drinking_report(row, instance=instance),
            "heart_function": self.get_heart_function(row, instance=instance),
            "MRI_taken_time": self.get_MRI_taken_time(row, instance=instance),
            # "physical_activity": self.get_physical_activity(row, instance=instance),
            # "ecg_diagnoses": self.get_ecg_diagnoses(row, instance=instance),
            # "parent_cardiac_diagnoses": self.get_parent_cardiac_diagnoses(row),
        }

def generate_report(
    metadata_df: pd.DataFrame,
    field_to_map: dict[int, dict[int, str]],
) -> pd.DataFrame:
    """Generate a report for the parsed metadata.

    Args:
        metadata_df: Parsed metadata DataFrame.
        field_to_map: Mapping of field ID to data coding.

    Returns:
        Report DataFrame.
    """
    parser = Parser(field_to_map)
    reports = []
    for _, row in tqdm(metadata_df.iterrows(), total=metadata_df.shape[0]):
        reports.append(parser.parse_row(row))
    return pd.DataFrame(reports)

def parse_args() -> argparse.Namespace:
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv_path",
        default=Path(".../text/ukb675874.csv"),
        type=Path,
        help="Path of UKB metadata csv.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Number of rows to read per chunk.",
    )
    parser.add_argument(
        "--out_dir",
        default=Path(".../text"),
        type=Path,
        help="Folder to save the downloaded files.",
    )
    parser.add_argument(
        "--background_template",
        default=Path(".../text/background_template.json"),
        type=Path,
        help="Folder to save the downloaded files.",
    )
    args = parser.parse_args()

    return args

def convert_days_to_text_history(time_dict):
    result_parts = []
    
    for condition, days in time_dict.items():
        years = days / 365
        if years >= 1:
            years = int(years)
            time_str = f"{years} year{'s' if years > 1 else ''}"
        else:
            months = int(days / 30)
            time_str = f"{months} month{'s' if months > 1 else ''}"
            
        result_parts.append(f"{condition} {time_str} ago")
    
    # Join all parts with commas and 'and' for the last item
    if len(result_parts) > 1:
        return ', '.join(result_parts[:-1]) + ' and ' + result_parts[-1]
    else:
        return result_parts[0]

def generate_background_info(row, templates):

    report_sections = []

    # Age and gender section
    if not pd.isna(row.get('age')) and not pd.isna(row.get('sex')):
        section = random.choice(templates['age_and_gender']) \
                 .replace('{age}', str(row['age'])) \
                 .replace('{gender}', row['sex'])
        report_sections.append(section)

    # BMI section 
    required_bmi_fields = ['height', 'weight', 'BMI']
    if all(not pd.isna(row.get(field)) for field in required_bmi_fields):
        section = random.choice(templates['BMI']) \
                 .replace('{height}', str(row['height'])) \
                 .replace('{weight}', str(row['weight'])) \
                 .replace('{BMI}', str(row['BMI']))
        report_sections.append(section)

    # Symptoms section
    if not pd.isna(row.get('symptoms')):
        section = random.choice(templates['symptoms']) \
                 .replace('{symptoms}', row['symptoms'])
        report_sections.append(section)

    # Vital signs section
    vital_fields = ['pulse_rate', 'systolic_blood_pressure', 'diastolic_blood_pressure']
    if all(not pd.isna(row.get(field)) for field in vital_fields):
        section = random.choice(templates['pulse_rate_and_systolic_diastolic']) \
                 .replace('{pulse_rate}', str(row['pulse_rate'])) \
                 .replace('{systolic}', str(row['systolic_blood_pressure'])) \
                 .replace('{diastolic}', str(row['diastolic_blood_pressure']))
        report_sections.append(section)

    # Cardiac history section
    if row.get('cardiac_history_with_time') != '{}':
        section = random.choice(templates['cardiac_history']) \
                 .replace('{diseases_with_time}', 
                         convert_days_to_text_history(ast.literal_eval(row['cardiac_history_with_time'])))
    else:
        section = random.choice(templates['no_cardiac_history'])
    report_sections.append(section)

    # Other medical history section
    if row.get('cardiac_related_history_with_time') != '{}':
        section = random.choice(templates['other_history']) \
                 .replace('{diseases_with_time}',
                         convert_days_to_text_history(ast.literal_eval(row['cardiac_related_history_with_time'])))
        report_sections.append(section)

    # Family history section
    if not pd.isna(row.get('family_history')):
        report_sections.append(row['family_history'].capitalize())

    # Lifestyle sections
    if not pd.isna(row.get('drinking')):
        report_sections.append(f"The patient is a {row['drinking']}")
        
    if not pd.isna(row.get('smoking')):
        report_sections.append(f"The patient is a {row['smoking']}")

    if not pd.isna(row.get('MET')) and not pd.isna(row.get('sleep_duration')):
        section = random.choice(templates['lifestyle']) \
                 .replace('{MET}', str(row['MET'])) \
                 .replace('{sleep_hour}', str(row['sleep_duration']))
        report_sections.append(section)

    return ' '.join(report_sections)

def generate_bg_data(df, background_templates):

    bgs = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        bg = generate_background_info(row, background_templates)
        bgs.append({
            "eid":row["eid"],
            "background":bg
        })

    return pd.DataFrame(bgs)

def main() -> None:
    """Main function."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # download data coding files for all fields
    field_df = pd.read_csv(Path(__file__).parent.resolve() / "fields.csv")
    field_to_map = get_field_to_map(field_df, args.out_dir)

    # select columns from the metadata csv
    ukb_sample_df = pd.read_csv(args.csv_path, nrows=2)
    field_ids = field_df["UDI"].astype(str).tolist()
    columns = [x for x in ukb_sample_df.columns if x.split("-")[0] in field_ids]
    metadata_df = read_csv_by_chunk(args.csv_path, columns, chunksize=args.chunksize)
    metadata_df.to_csv(args.out_dir / "selected_metadata.csv", index=False)

    # parse the selected metadata to generate a report
    metadata_df = pd.read_csv(args.out_dir / "selected_metadata.csv", low_memory=False)
    report_df = generate_report(metadata_df, field_to_map)
    report_df.to_csv(args.out_dir / "report.csv", index=False)

    # generate the background based on the report
    report_df = pd.read_csv(args.out_dir / "report.csv", low_memory=False)
    with open(args.background_template) as f:
        background_templates = json.load(f)
    bg_df = generate_bg_data(report_df, background_templates)
    bg_df.to_csv(args.out_dir / "background.csv", index=False)

if __name__ == "__main__":
    main()