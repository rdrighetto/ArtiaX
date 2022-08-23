# vim: set expandtab shiftwidth=4 softtabstop=4:

# General
import numpy as np
import starfile
import pandas as pd

# Chimerax
from chimerax.core.errors import UserError

# This package
from ..formats import ArtiaXFormat
from ..ParticleData import ParticleData, EulerRotation

EPSILON = np.finfo(np.float32).eps
EPSILON16 = 16 * EPSILON

class STOPGAPEulerRotation(EulerRotation):

    def __init__(self):
        super().__init__(axis_1=(0, 0, 1), axis_2=(1, 0, 0), axis_3=(0, 0, 1))

    def rot1_from_matrix(self, matrix):
        """Phi"""
        # Singularity check
        if matrix[2, 2] > 0.9999:
            angle = 0
        else:
            angle = np.arctan2(matrix[2, 0], matrix[2, 1]) * 180.0 / np.pi

        return angle

    def rot2_from_matrix(self, matrix):
        """Theta"""
        angle = np.arctan2(np.sqrt(1 - (matrix[2, 2] * matrix[2, 2])), matrix[2, 2]) * 180.0 / np.pi

        return angle

    def rot3_from_matrix(self, matrix):
        """Psi"""
        # Singularity check
        if matrix[2, 2] > 0.9999:
            angle = -1.0 * np.sign(matrix[0, 1]) * np.arccos(matrix[0, 0]) * 180.0/np.pi
        else:
            angle = np.arctan2(matrix[0, 2], -matrix[1, 2]) * 180.0 / np.pi

        return angle




class STOPGAPParticleData(ParticleData):

    DATA_KEYS = {
        'motl_idx': [],
        'tomo_num': [],
        'object': [],
        'subtomo_num': [],
        'halfset': [],
        'orig_x': [],
        'orig_y': [],
        'orig_z': [],
        'score': [],
        'x_shift': [],
        'y_shift': [],
        'z_shift': [],
        'phi': [],
        'psi': [],
        'the': [],
        'class': []
    }

    DEFAULT_PARAMS = {
        'pos_x': 'orig_x',
        'pos_y': 'orig_y',
        'pos_z': 'orig_z',
        'shift_x': 'x_shift',
        'shift_y': 'y_shift',
        'shift_z': 'z_shift',
        'ang_1': 'phi',
        'ang_2': 'the',
        'ang_3': 'psi'
    }

    ROT = STOPGAPEulerRotation

    def __init__(self, session, file_name, oripix=1, trapix=1, additional_files=None):
        self.remaining_loops = {}
        self.remaining_data = {}
        self.loop_name = 0
        self.name_prefix = None
        self.name_leading_zeros = None

        super().__init__(session, file_name, oripix=oripix, trapix=trapix, additional_files=additional_files)

    def read_file(self):
        content = starfile.read(self.file_name, always_dict=True)

        # Identify the loop that contains the data
        data_loop = None
        for key, val in content.items():
            if 'orig_z' in list(val.keys()):
                data_loop = key
                break

        # Abort if none found
        if data_loop is None:
            raise UserError('orig_z was not found in any loop section of file {}.'.format(self.file_name))

        # Take the good one, store the rest and the loop name so we can write it out again later on
        df = content[data_loop]
        content.pop(data_loop)
        self.loop_name = data_loop
        self.remaining_loops = content

        # What is present
        df_keys = list(df.keys())

        # Store everything
        self._register_keys()

        # Now make particles
        df.reset_index()


        for idx, row in df.iterrows():
            p = self.new_particle()

            p['motl_idx'] = row['motl_idx']
            p['tomo_num'] = row['tomo_num']
            p['object'] = row['object']
            p['subtomo_num'] = row['subtomo_num']
            p['halfset'] = row['halfset']
            p['orig_x'] = row['orig_x']-1
            p['orig_y'] = row['orig_y']-1
            p['orig_z'] = row['orig_z']-1
            p['score'] = row['score']
            p['x_shift'] = row['x_shift']
            p['y_shift'] = row['y_shift']
            p['z_shift'] = row['z_shift']
            p['phi'] = row['phi']
            p['psi'] = row['psi']
            p['the'] = row['the']
            p['class'] = row['class']


    # def write_file(self, file_name=None, additional_files=None):
    # TO-DO:
    # -- Implement writing of STOPGAP STAR files offsetting the origin coordinates by 1 (see ArtiatomiParticleData.py)
    # -- Implement correct STAR file header for STOPGAP motivelists (currently writes RELION STAR header)
    # RDR 2022-08-23
    # 
    #     if file_name is None:
    #         file_name = self.file_name

    #     data = self.as_dictionary()

    #     df = pd.DataFrame(data=data)

    #     full_dict = self.remaining_loops
    #     full_dict[self.loop_name] = df

    #     starfile.write(full_dict, file_name, overwrite=True)

STOPGAP_FORMAT = ArtiaXFormat(name='STOPGAP STAR file',
                             nicks=['stopgap', 'star', 'motl', 'motivelist'],
                             particle_data=STOPGAPParticleData)