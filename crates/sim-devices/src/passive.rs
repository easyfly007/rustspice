#[derive(Debug, Clone)]
pub enum PassiveKind {
    Resistor,
    Capacitor,
    Inductor,
}

#[derive(Debug, Clone)]
pub struct PassiveDevice {
    pub name: String,
    pub kind: PassiveKind,
}
