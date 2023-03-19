use rand::Rng;

mod minitorch;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;

    pub fn assert_close(a: f64, b: f64) {
        assert!(minitorch::operators::is_close(a, b) != 0.0, "Failure! x={} y={}", a, b)
    }

    pub fn gen_float() -> f64 {
        rand::thread_rng().gen_range(-100.0..100.0)
    }
}
