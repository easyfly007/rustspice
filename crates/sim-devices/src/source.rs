#[derive(Debug, Clone)]
pub enum SourceKind {
    Voltage,
    Current,
}

#[derive(Debug, Clone)]
pub struct SourceDevice {
    pub name: String,
    pub kind: SourceKind,
}
