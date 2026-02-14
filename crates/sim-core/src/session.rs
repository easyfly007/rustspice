#[derive(Debug, Clone)]
pub enum SessionState {
    Parsed,
    Elaborated,
    Ready,
    Running,
    Completed,
}

#[derive(Debug, Clone)]
pub struct Session {
    pub state: SessionState,
}

impl Session {
    pub fn new() -> Self {
        Self {
            state: SessionState::Parsed,
        }
    }
}

pub fn debug_dump_session(session: &Session) {
    println!("session: state={:?}", session.state);
}
