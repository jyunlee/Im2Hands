# from artihand.nasa.models.core import ArticulatedHandNet HaloVAE
from models.naive.models.core import HaloVAE

from models.naive.models import decoder
from models.naive.models import encoder
from models.naive.models.refine import RefineNet

# Encoder latent dictionary
# encoder_latent_dict = {
#     'simple': encoder_latent.Encoder,
# }

encoder_dict = {
    'simple': encoder.SimpleEncoder,
    'pointnet': encoder.ResnetPointnet
}

encoder_latent_dict = {
    'simple_latent': encoder.LetentEncoder,
}

# Decoder dictionary
decoder_dict = {
    'simple': decoder.SimpleDecoder,
    # 'piece_rigid': decoder.PiecewiseRigidDecoder,
    # 'piece_deform': decoder.PiecewiseDeformableDecoder,
}
