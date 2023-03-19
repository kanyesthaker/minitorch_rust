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

pub fn inv_back(x: f64, d: f64) -> f64 {
    return -d/mul(x, x)
}

pub fn is_close(x: f64, y: f64) -> f64 {
    (f64::abs(x-y) < 1e-2) as i64 as f64
}

pub fn relu(x: f64) -> f64 {
    f64::max(x, 0.0)
}

pub fn relu_back(x: f64, d: f64) -> f64 {
    if x > 0.0 {
        return d;
    }
    return 0.0
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        return 1.0 / (1.0 + f64::exp(-x))
    }
    f64::exp(x)/(1.0 + f64::exp(x))
}

pub fn log(x: f64) -> f64 {
    if x == 0.0 {
        return 1e-6_f64.ln();
    }
    x.ln()
}

pub fn log_back(x: f64, d: f64) -> f64 {
    mul(inv(x), d)
}

pub fn exp(x: f64) -> f64 {
    f64::exp(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::assert_close;
    use crate::test_utils::gen_float;
    use rstest::rstest;

    #[rstest]
    fn test_basic() {
        assert_close(mul(1.0, 2.0), 2.0);
        assert_close(add(1.0, 2.0), 3.0);
        assert_close(neg(1.0), -1.0);
        assert_close(max(0.0, 1.0), 1.0);
        assert_close(max(1.0, 1.0), 1.0);
        assert_close(inv(2.0), 0.5);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_relu(#[case] input: f64) {
        if input > 0.0 {
            assert!(relu(input) == input);
        } else {
            assert!(relu(input) == 0.0);
        }
    }

    #[rstest]
    #[case(gen_float())]
    fn test_id(#[case] input: f64) {
        assert!(id(input) == input);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_lt(#[case] input: f64) {
        assert!(lt(input, input - 1.0) == 0.0);
        assert!(lt(input - 1.0, input) == 1.0);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_max(#[case] input: f64) {
        assert!(max(input, input - 1.0) == input);
        assert!(max(input - 1.0, input) == input);
        assert!(max(input, input) == input);
    }

    #[rstest]
    #[case(gen_float())]
    fn test_eq(#[case] input: f64) {
        assert!(eq(input, input) == 1.0);
        assert!(eq(input, input + 1.0) == 0.0);
        assert!(eq(input + 1.0, input) == 0.0);
    }

    #[rstest]
    #[case(gen_float(), gen_float())]
    fn test_relu_back(#[case] input: f64, #[case] constant: f64) {
        if input > 0.0 {
            assert!(relu_back(input, constant) == constant);
        } else if input < 0.0 {
            assert!(relu_back(input, constant) == 0.0);
        }
    }

    #[rstest]
    #[case(gen_float())]
    fn test_sigmoid(#[case] input: f64) {
        assert!(0.0 <= sigmoid(input) && sigmoid(input) <= 1.0);
        assert_close(1.0 - sigmoid(input), sigmoid(-input));
        assert_close(sigmoid(0.0), 0.5);
        assert!(
            sigmoid(input + 1.0) >= sigmoid(input) 
            && sigmoid(input) >= sigmoid(input - 1.0)
            && sigmoid(input - 1.0)>= 0.0
        );
    }
}
