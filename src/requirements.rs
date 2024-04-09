use crate::{
    error::{Error, Result},
    types::{Band, PMParameters, ParametersBuilder},
};
use num_traits::Float;

/// Creates parameters for the Parks-McClellan algorithm in terms of a list of
/// [`BandSetting`]s.
///
/// This function is used by one of the two coding styles supported by this
/// crate. It uses a list of `BandSetting`s to specify each of the bands and
/// its desired function and weight used in the Parks-McClellan algorithm. The
/// `num_taps` parameter indicates the number of taps of FIR filter to be
/// designed.
///
/// As in [`PMParameters`], some of the parameters required for the
/// Parks-McClellan algorith are given default values (the same defaults as in
/// `PMParameters`, and in particular, an even symmetry for the FIR taps). The
/// defaults can be changed by using the methods defined by the
/// [`ParametersBuilder`] trait, which is implemented by the object returned by
/// this function.
///
/// In technical terms, this function constructs and returns an appropriate
/// `PMParameters` object. However, the type parameters `D` and `W` of that
/// object cannot be named, because they are closures with anonymous
/// types. Therefore, an `impl ParametersBuilder` is used as the return type of
/// the function.
pub fn pm_parameters<T: Float>(
    num_taps: usize,
    band_settings: &[BandSetting<T>],
) -> Result<impl ParametersBuilder<T> + '_> {
    if band_settings.is_empty() {
        return Err(Error::BandsEmpty);
    }
    let bands = band_settings.iter().map(|setting| setting.band()).collect();
    let desired = move |f| desired_response(band_settings, f);
    let weights = move |f| weight(band_settings, f);
    PMParameters::new(num_taps, bands, desired, weights)
}

fn desired_response<T: Float>(settings: &[BandSetting<T>], freq: T) -> T {
    let setting = closest_setting(settings, freq);
    setting.desired_response.evaluate(freq, &setting.band)
}

fn weight<T: Float>(settings: &[BandSetting<T>], freq: T) -> T {
    let setting = closest_setting(settings, freq);
    setting.weight.evaluate(freq, &setting.band)
}

fn closest_setting<T: Float>(settings: &[BandSetting<T>], freq: T) -> &BandSetting<T> {
    settings
        .iter()
        .min_by(|a, b| {
            a.band
                .distance(freq)
                .partial_cmp(&b.band.distance(freq))
                .unwrap()
        })
        .unwrap()
}

/// Band with desired response and weight [`Setting`]s.
///
/// This struct defines a band (a closed subinterval of [0.0, 0.5] in which the
/// Parks-McClellan algorithm tries to minimize the maximum weighted error) to
/// which a desired response and a weight function in the form of [`Setting`]s
/// are attached. The struct is used in one of the coding styles supported by
/// this crate. In such coding style, the Parks-McClellan algorithm parameters
/// are defined in terms of a list of `BandSetting`s by calling the
/// [`pm_parameters`] function.
#[derive(Debug)]
pub struct BandSetting<T> {
    band: Band<T>,
    desired_response: Setting<T>,
    weight: Setting<T>,
}

impl<T: Float> BandSetting<T> {
    /// Creates a new `BandSetting` with default weight.
    ///
    /// The `band_begin` and `band_end` parameter indicate the being and the end
    /// of the band respectively. The `desired_response` parameter gives the
    /// desired response in this band. The weight when using this constructor is
    /// set to `constant(T::one())`. A custom weight can be defined using the
    /// constructor [`BandSetting::with_weight`] instead, or by calling
    /// [`BandSetting::set_weight`].
    pub fn new(band_begin: T, band_end: T, desired_response: Setting<T>) -> Result<BandSetting<T>> {
        let weight = constant(T::one());
        BandSetting::with_weight(band_begin, band_end, desired_response, weight)
    }

    /// Creates a new `BandSetting` with a custom weight.
    ///
    /// The `weight` parameter gives the weight function in this band. The
    /// remaining parameters behave as in [`BandSetting::new`].
    pub fn with_weight(
        band_begin: T,
        band_end: T,
        desired_response: Setting<T>,
        weight: Setting<T>,
    ) -> Result<BandSetting<T>> {
        let band = Band::new(band_begin, band_end)?;
        Ok(BandSetting {
            band,
            desired_response,
            weight,
        })
    }

    /// Returns the [`Band`] associated to this [`BandSetting`].
    pub fn band(&self) -> Band<T> {
        self.band
    }

    /// Sets the [`Band`] associated to this [`BandSetting`].
    pub fn set_band(&mut self, band: Band<T>) {
        self.band = band;
    }

    /// Sets the desired response used by this [`BandSetting`].
    pub fn set_desired_response(&mut self, desired_response: Setting<T>) {
        self.desired_response = desired_response;
    }

    /// Sets the weight function used by this [`BandSetting`].
    pub fn set_weight(&mut self, weight: Setting<T>) {
        self.weight = weight
    }
}

/// Desired response or weight setting.
///
/// This struct is used to indicate the desired response or the weigth function
/// for a band through a [`BandSetting`] object when using the coding style that
/// employs the [`pm_parameters`] function to indicate the Parks-McClellan
/// algorithm parameters in terms of a list of [`BandSetting`]s.
///
/// Values of this object are constructed using the [`constant`], [`linear`],
/// and [`function`] functions, which create a [`Setting`] that represents a
/// constant function, a linear function, or a function defined by an arbitrary
/// closure respectively.
#[derive(Debug)]
pub struct Setting<T>(SettingData<T>);

impl<T: Float> Setting<T> {
    fn evaluate(&self, x: T, band: &Band<T>) -> T {
        match &self.0 {
            SettingData::Constant { value } => *value,
            SettingData::Linear { begin, end } => {
                let u = (x - band.begin()) / (band.end() - band.begin());
                *begin + u * (*end - *begin)
            }
            SettingData::Function { f } => (f)(x),
        }
    }
}

enum SettingData<T> {
    Constant { value: T },
    Linear { begin: T, end: T },
    Function { f: Box<dyn Fn(T) -> T> },
}

/// Creates a [`Setting`] that represents a constant function.
pub fn constant<T: Float>(value: T) -> Setting<T> {
    Setting(SettingData::Constant { value })
}

/// Creates a [`Setting`] that represents a linear function.
///
/// The function has the values `begin` and `end` at the begin and end of the
/// band to which the `Setting` is applied respectively, and it is linearly
/// interpolated for the remaining points of the band.
pub fn linear<T: Float>(begin: T, end: T) -> Setting<T> {
    Setting(SettingData::Linear { begin, end })
}

/// Creates a [`Setting`] that represents an arbitrary function.
///
/// The arbitrary function is provided as a boxed closure trait object. The
/// closure will only be evaluated at points belonging to the band to which the
/// `Setting` is applied.
pub fn function<T: Float>(f: Box<dyn Fn(T) -> T>) -> Setting<T> {
    Setting(SettingData::Function { f })
}

impl<T: std::fmt::Debug> std::fmt::Debug for SettingData<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            SettingData::Constant { value } => {
                f.debug_struct("Constant").field("value", value).finish()
            }
            SettingData::Linear { begin, end } => f
                .debug_struct("Linear")
                .field("begin", begin)
                .field("end", end)
                .finish(),
            SettingData::Function { .. } => f.debug_struct("Function").finish(),
        }
    }
}
