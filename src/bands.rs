use crate::{
    error::{Error, Result},
    types::Band,
};
use num_traits::Float;

// Check that the bands are correctly defined
pub fn check_bands<T: Float>(bands: &[Band<T>]) -> Result<()> {
    if bands.is_empty() {
        return Err(Error::BandsEmpty);
    }
    for (j, band1) in bands.iter().enumerate() {
        for band2 in bands.iter().skip(j + 1) {
            if band1.overlaps(band2) {
                return Err(Error::BandsOverlap);
            }
        }
    }

    Ok(())
}

// Sort bands in increasing order
pub fn sort_bands<T: Float>(bands: &[Band<T>]) -> Vec<Band<T>> {
    let mut bands = bands.to_vec();
    bands.sort_unstable_by(|a, b| a.begin().partial_cmp(&b.begin()).unwrap());
    bands
}
