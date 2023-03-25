use std::any::Any;

pub trait RequiresGrad {
    fn requires_grad_(&self, bool) -> ();
    fn set_name(&self, String) -> ();
    fn name(&self) -> &str;
}

pub struct Variable {
    name: Option<String>,
    requires_grad: bool,
}


pub struct Parameter {
    // The type here is Box<dyn Any>. Box stores the value on the heap instead of the stack
    // and the dyn keyword allows us to downcast `value` at runtime, and use dynamic dispatch
    value: &impl Variable,
    name: Option<String>,
}

impl Parameter {
    // The `new` trait is like Python's `__init__`
    pub fn new(x: &impl Variable, name:Option<String>) -> Self {
        let mut parameter = Parameter {
            value: &x,
            name: name,
        };
        parameter.value.requires_grad_(true);
            req_grad.call(true);
            if let Some(name) = &parameter.name {
                parameter.*value.name = name.clone();
            }
        }
        parameter
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    pub struct TestVariable {

    }
    #[rstest]
    fn test_basic_parameter() {
        let mut p = Parameter::new(42, None);
        assert!(*p.value == 42);
    }
}
