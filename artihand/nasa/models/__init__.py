from artihand.nasa.models.core_init_occ import ArticulatedHandNetInitOcc
from artihand.nasa.models.core_ref_occ import ArticulatedHandNetRefOcc
from artihand.nasa.models.core_kpts_ref import ArticulatedHandNetKptsRef
from artihand.nasa.models import decoder

decoder_dict = {
    'simple': decoder.SimpleDecoder,
    'piece_rigid': decoder.PiecewiseRigidDecoder,
    'piece_deform': decoder.PiecewiseDeformableDecoder,
    'piece_deform_pifu': decoder.PiecewiseDeformableDecoderPIFu,
    'sdf_simple': decoder.SdfDecoder
}
