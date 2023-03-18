pub fn mul(x: f64, y: f64) -> f64 {
    x * y
}

pub fn id(x: f64) -> f64  {
    x
}

pub fn add(x: f64, y: f64) -> f64 {
    x + y
}

pub fn neg(x: f64) -> f64 {
    -x
}

pub fn lt(x: f64, y:f64) -> f64 {
    (x < y) as i64 as f64
}

pub fn eq(x: f64, y: f64) -> f64 {
    (x == y) as i64 as f64
}

pub fn max(x: f64, y: f64) -> f64 {
    f64::max(y, x)
}

pub fn inv(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    1.0 / x
}

pub fn is_close(x: f64, y: f64) -> f64 {
    (f64::abs(x-y) < 1e-2) as i64 as f64
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        return 1.0 / (1.0 + f64::exp(-x))
    }
    f64::exp(x)/(1.0 + f64::exp(x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_close;

    #[test]
    fn test_basic() {
        assert_close(mul(1.0, 2.0), 2.0);
        assert_close(add(1.0, 2.0), 3.0);
        assert_close(neg(1.0), -1.0);
        assert_close(max(0.0, 1.0), 1.0);
        assert_close(max(1.0, 1.0), 1.0);
        assert_close(inv(2.0), 0.5);
    }


}
