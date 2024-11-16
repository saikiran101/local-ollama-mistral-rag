// import React, { useState, useRef, useEffect } from 'react';
// import axios from 'axios';
// import { Container, Row, Col, Form, Button, Card, Modal, Badge } from 'react-bootstrap';
// import './App.css';

// function App() {
//   const [input, setInput] = useState('');
//   const [messages, setMessages] = useState([]);
//   const [isLoading, setIsLoading] = useState(false);
//   const [showModal, setShowModal] = useState(false);
//   const [modalContent, setModalContent] = useState('');
//   const [modalTitle, setModalTitle] = useState('');
//   const messagesEndRef = useRef(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(scrollToBottom, [messages]);

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!input.trim()) return;

//     setIsLoading(true);
//     setMessages(prev => [...prev, { text: input, sender: 'user' }]);
//     setInput('');

//     try {
//       const response = await axios.post('http://localhost:8000/ask', { text: input });
//       setMessages(prev => [...prev, { 
//         text: response.data.answer, 
//         sender: 'bot', 
//         confidence: response.data.confidence,
//         pageNumbers: response.data.page_numbers
//       }]);
//     } catch (error) {
//       console.error('Error:', error);
//       setMessages(prev => [...prev, { text: 'Sorry, an error occurred.', sender: 'bot' }]);
//     }

//     setIsLoading(false);
//   };

//   const handlePageClick = async (pageNumber) => {
//     try {
//       // Ensure pageNumber is always at least 1
//       const adjustedPageNumber = Math.max(1, pageNumber);
//       const response = await axios.get(`http://localhost:8000/get_page/${adjustedPageNumber}`);
//       setModalContent(response.data.page_text);
//       setModalTitle(`Page ${adjustedPageNumber}`);
//       setShowModal(true);
//     } catch (error) {
//       console.error('Error fetching page content:', error);
//       setModalContent('Error: Unable to fetch page content. ' + error.response?.data?.detail || error.message);
//       setModalTitle('Error');
//       setShowModal(true);
//     }
//   };

//   return (
//     <Container fluid className="d-flex flex-column vh-100 bg-light">
//       <Row className="bg-dark text-white py-3 mb-4">
//         <Col>
//           <h2 className="text-center">Mistral 8B Model Chatbot</h2>
//         </Col>
//       </Row>
//       <Row className="flex-grow-1 overflow-hidden">
//         <Col md={{ span: 8, offset: 2 }}>
//           <Card className="h-100">
//             <Card.Body className="chat-container">
//               {messages.map((message, index) => (
//                 <div key={index} className={`d-flex ${message.sender === 'user' ? 'justify-content-end' : 'justify-content-start'} mb-3`}>
//                   <div className={`p-2 rounded-3 ${message.sender === 'user' ? 'bg-primary text-white' : 'bg-light'} chat-message`}>
//                     <p className="mb-0">{message.text}</p>
//                     {message.confidence && <small className="d-block mt-1 text-muted">Confidence: {(message.confidence * 100).toFixed(2)}%</small>}
                
//                     {message.pageNumbers && (
//                       <div className="mt-2">
//                         {message.pageNumbers.map((pageNum) => (
//                           <Badge 
//                             key={pageNum} 
//                             bg="secondary" 
//                             className="me-1 clickable-badge" 
//                             onClick={() => handlePageClick(pageNum)}
//                           >
//                             Page {pageNum}
//                           </Badge>
//                         ))}
//                       </div>
//                     )}
//                   </div>
//                 </div>
//               ))}
//               {isLoading && (
//                 <div className="d-flex justify-content-start mb-3">
//                   <div className="p-2 rounded-3 bg-light chat-message">
//                     <p className="mb-0">Thinking...</p>
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRef} />
//             </Card.Body>
//           </Card>
//         </Col>
//       </Row>
//       <Row className="mt-4 mb-3">
//         <Col md={{ span: 8, offset: 2 }}>
//           <Form onSubmit={handleSubmit}>
//             <Form.Group as={Row} className="mb-3">
//               <Col sm={10}>
//                 <Form.Control
//                   type="text"
//                   value={input}
//                   onChange={(e) => setInput(e.target.value)}
//                   placeholder="Ask a question..."
//                   disabled={isLoading}
//                 />
//               </Col>
//               <Col sm={2}>
//                 <Button type="submit" disabled={isLoading} className="w-100">
//                   Send
//                 </Button>
//               </Col>
//             </Form.Group>
//           </Form>
//         </Col>
//       </Row>

//       <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
//         <Modal.Header closeButton>
//           <Modal.Title>{modalTitle}</Modal.Title>
//         </Modal.Header>
//         <Modal.Body>
//           <pre>{modalContent}</pre>
//         </Modal.Body>
//       </Modal>
//     </Container>
//   );
// }

// export default App;
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Container, Row, Col, Form, Button, Card, Modal, Badge } from 'react-bootstrap';
import './App.css';

function App() {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showModal, setShowModal] = useState(false);
  const [modalContent, setModalContent] = useState('');
  const [modalTitle, setModalTitle] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(scrollToBottom, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    setIsLoading(true);
    const newMessages = [...messages, { text: input, sender: 'user' }];
    setMessages(newMessages);
    setInput('');

    try {
      const response = await axios.post('http://localhost:8000/ask', { text: input });
      if (response.data && response.data.answer) {
        const formattedAnswer = formatAnswer(response.data.answer);
        setMessages(prev => [...prev, { 
          text: formattedAnswer, 
          sender: 'bot', 
          confidence: response.data.confidence,
          pageNumbers: response.data.page_numbers
        }]);
      } else {
        throw new Error("Invalid response format");
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, { text: 'Sorry, an error occurred.', sender: 'bot' }]);
    }

    setIsLoading(false);
  };

  const handlePageClick = async (pageNumber) => {
    try {
      const adjustedPageNumber = Math.max(1, pageNumber);
      const response = await axios.get(`http://localhost:8000/get_page/${adjustedPageNumber}`);
      setModalContent(response.data.page_text);
      setModalTitle(`Page ${adjustedPageNumber}`);
      setShowModal(true);
    } catch (error) {
      console.error('Error fetching page content:', error);
      setModalContent('Error: Unable to fetch page content. ' + error.response?.data?.detail || error.message);
      setModalTitle('Error');
      setShowModal(true);
    }
  };

  const formatAnswer = (answer) => {
    return answer.split('\n').map((line, index) => <p key={index}>{line}</p>);
  };

  return (
    <Container fluid className="d-flex flex-column vh-100 bg-light">
      <Row className="bg-dark text-white py-3 mb-4">
        <Col>
          <h2 className="text-center">Mistral 8B Model Chatbot</h2>
        </Col>
      </Row>
      <Row className="flex-grow-1 overflow-hidden">
        <Col md={{ span: 8, offset: 2 }}>
          <Card className="h-100">
            <Card.Body className="chat-container">
              {messages.map((message, index) => (
                <div key={index} className={`d-flex ${message.sender === 'user' ? 'justify-content-end' : 'justify-content-start'} mb-3`}>
                  <div className={`p-2 rounded-3 ${message.sender === 'user' ? 'bg-primary text-white' : 'bg-light'} chat-message`}>
                    {Array.isArray(message.text) ? message.text.map((line, idx) => (
                      <div key={idx}>{line}</div>
                    )) : message.text}
                    {message.confidence && <small className="d-block mt-1 text-muted">Confidence: {(message.confidence * 100).toFixed(2)}%</small>}
                    {message.pageNumbers && (
                      <div className="mt-2">
                        {message.pageNumbers.map((pageNum) => (
                          <Badge 
                            key={pageNum} 
                            bg="secondary" 
                            className="me-1 clickable-badge" 
                            onClick={() => handlePageClick(pageNum)}
                          >
                            Page {pageNum}
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="d-flex justify-content-start mb-3">
                  <div className="p-2 rounded-3 bg-light chat-message">
                    <p className="mb-0">Thinking...</p>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </Card.Body>
          </Card>
        </Col>
      </Row>
      <Row className="mt-4 mb-3">
        <Col md={{ span: 8, offset: 2 }}>
          <Form onSubmit={handleSubmit}>
            <Form.Group as={Row} className="mb-3">
              <Col sm={10}>
                <Form.Control
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask a question..."
                  disabled={isLoading}
                />
              </Col>
              <Col sm={2}>
                <Button type="submit" disabled={isLoading} className="w-100">
                  Send
                </Button>
              </Col>
            </Form.Group>
          </Form>
        </Col>
      </Row>

      <Modal show={showModal} onHide={() => setShowModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>{modalTitle}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <pre>{modalContent}</pre>
        </Modal.Body>
      </Modal>
    </Container>
  );
}

export default App;
