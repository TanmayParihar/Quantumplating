from qiskit import transpile, QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler, Session
import numpy as np
import qiskit
print(qiskit.__version__)

class QuantumErrorCorrection:
    def __init__(self, api_token=None):

        self.service = QiskitRuntimeService(channel="ibm_quantum", token=api_token)

        available_backends = self.service.backends()
        print("Available backends:", [backend.name for backend in available_backends])

        self.backend = self.service.least_busy(simulator=False)
        print(f"Selected backend: {self.backend.name}")
        print(f"Backend status: {self.backend.status()}")

    def create_bit_flip_code(self, error_probability=0.3):

        q_data = QuantumRegister(1, 'q_data')
        q_ancilla = QuantumRegister(2, 'q_ancilla')
        c_syndrome = ClassicalRegister(2, 'c_syndrome')
        c_output = ClassicalRegister(1, 'c_output')

        qc = QuantumCircuit(q_data, q_ancilla, c_syndrome, c_output)

        # Prepare initial state (|0⟩ + |1⟩)/√2
        qc.h(q_data[0])

        # Encoding
        qc.cx(q_data[0], q_ancilla[0])
        qc.cx(q_data[0], q_ancilla[1])

        # Add artificial noise (bit flip) with probability
        qc.rx(error_probability * np.pi, q_data[0])

        # Error detection
        qc.cx(q_data[0], q_ancilla[0])
        qc.cx(q_data[0], q_ancilla[1])

        # Measure syndrome
        qc.measure(q_ancilla, c_syndrome)

        # Error correction based on syndrome
        with qc.if_test((c_syndrome, 1)):  # If syndrome is 01
            qc.x(q_data[0])
        with qc.if_test((c_syndrome, 2)):  # If syndrome is 10
            qc.x(q_data[0])

        # Measure final state
        qc.measure(q_data, c_output)

        return qc

    def create_phase_flip_code(self):

        q_data = QuantumRegister(1, 'q_data_p')
        q_ancilla = QuantumRegister(2, 'q_ancilla_p')
        c_syndrome = ClassicalRegister(2, 'c_syndrome_p')
        c_output = ClassicalRegister(1, 'c_output_p')

        qc = QuantumCircuit(q_data, q_ancilla, c_syndrome, c_output)

        # Initialize in superposition
        qc.h(q_data[0])

        # Encode in X-basis
        qc.h(q_data[0])
        qc.h(q_ancilla[0])
        qc.h(q_ancilla[1])
        qc.cx(q_data[0], q_ancilla[0])
        qc.cx(q_data[0], q_ancilla[1])

        # Simulate phase error
        qc.z(q_data[0])

        # Error detection
        qc.cx(q_data[0], q_ancilla[0])
        qc.cx(q_data[0], q_ancilla[1])
        qc.h(q_data[0])
        qc.h(q_ancilla[0])
        qc.h(q_ancilla[1])

        # Measure syndrome
        qc.measure(q_ancilla, c_syndrome)

        # Correct error based on syndrome
        with qc.if_test((c_syndrome, 1)):  # If syndrome is 01
            qc.z(q_data[0])

        # Ensure final measurement of the data qubit
        qc.measure(q_data, c_output)

        return qc

    def run_circuit(self, circuit, shots=1000):
        try:
            print(f"Submitting job to {self.backend.name}...")

            # Transpile the circuit for the target backend
            transpiled_circuit = transpile(circuit, backend=self.backend, optimization_level=3)

            with Session(backend=self.backend) as session:
                sampler = Sampler(mode=session)
                job = sampler.run([transpiled_circuit], shots=shots)
                print(f"Job ID: {job.job_id()}")
                print("Waiting for job results...")

                result = job.result()

                # Inspect the raw data for analysis
                print("Inspecting raw results:")
                if hasattr(result, '_pub_results'):
                    print("Raw result contents:", result._pub_results)

                    # Extract the first result from the list
                    raw_result = result._pub_results[0]  # Access the first SamplerPubResult
                    if hasattr(raw_result, 'data'):
                        if hasattr(raw_result.data, 'c_output'):
                            bitarray = raw_result.data.c_output
                        elif hasattr(raw_result.data, 'c_output_p'):
                            bitarray = raw_result.data.c_output_p
                        else:
                            raise ValueError("No valid c_output or c_output_p data found in raw results.")

                        # Use the get_counts method to extract measurement counts
                        counts = bitarray.get_counts()
                        print("Measurement counts:", counts)
                        return counts, self.backend


                    else:
                        raise ValueError("No valid c_output data found in raw results.")

                else:
                    raise AttributeError("Result object does not contain '_pub_results'.")

        except Exception as e:
            print(f"Error running on hardware: {e}")
            raise

    def analyze_results(self, counts, code_type):
        total_shots = sum(counts.values())
        # For bit flip code, success is measuring the correct original state
        success_rate = counts.get('0', 0) / total_shots * 100
        error_rate = 100 - success_rate

        analysis = {
            'code_type': code_type,
            'total_shots': total_shots,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'raw_counts': counts
        }

        return analysis


def main():
    API_TOKEN = "here"  # Replace with your actual API token
    qec = QuantumErrorCorrection(api_token=API_TOKEN)

    # Test bit flip code
    print("\nRunning bit flip code...")
    bit_flip_circuit = qec.create_bit_flip_code(error_probability=0.3)
    bit_flip_counts, bf_backend = qec.run_circuit(bit_flip_circuit)
    bit_flip_analysis = qec.analyze_results(bit_flip_counts, 'bit_flip')

    # Test phase flip code
    print("\nRunning phase flip code...")
    phase_flip_circuit = qec.create_phase_flip_code()
    phase_flip_counts, pf_backend = qec.run_circuit(phase_flip_circuit)
    phase_flip_analysis = qec.analyze_results(phase_flip_counts, 'phase_flip')

    # Print results
    print("\nResults Summary:")
    print(f"Bit Flip Code (on {bf_backend}):")
    print(f"Success Rate: {bit_flip_analysis['success_rate']:.2f}%")
    print(f"Error Rate: {bit_flip_analysis['error_rate']:.2f}%")
    print(f"Raw Counts: {bit_flip_analysis['raw_counts']}")

    print(f"\nPhase Flip Code (on {pf_backend}):")
    print(f"Success Rate: {phase_flip_analysis['success_rate']:.2f}%")
    print(f"Error Rate: {phase_flip_analysis['error_rate']:.2f}%")
    print(f"Raw Counts: {phase_flip_analysis['raw_counts']}")


if __name__ == "__main__":
    main()
